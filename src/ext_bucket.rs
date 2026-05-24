//! Disk-spilling bucket for the external-memory SA construction path.
//!
//! Port of `Ext_Mem_Bucket.hpp` from upstream CaPS-SA. A bucket holds a
//! `Vec`-like ordered sequence of fixed-size records, transparently spilling
//! to a backing file once the in-memory buffer fills. Records are written
//! and read back in insertion order.
//!
//! The bucket additionally tracks **sub-subarray boundaries** so that the
//! caller can append several runs (each one a sorted sub-subarray from a
//! different worker) and later recover their boundaries for a multi-way
//! merge.
//!
//! Generic over the record type via [`BucketRecord`], which provides
//! fixed-size little-endian serialization.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tempfile::NamedTempFile;

/// A fixed-size, byte-serializable record type stored in [`ExtMemBucket`].
///
/// Implementations of this trait must encode a value to a fixed-size,
/// architecture-independent byte representation (little-endian).
pub trait BucketRecord: Copy + Send + Sync {
    /// Number of bytes per record on disk.
    const SIZE: usize;
    /// Serialize one record into a byte slice of length [`Self::SIZE`].
    fn write_to(&self, out: &mut [u8]);
    /// Deserialize one record from a byte slice of length [`Self::SIZE`].
    fn read_from(bytes: &[u8]) -> Self;
}

/// Raw `u64` record — used by the heap-merge external-memory path where
/// no LCP information is carried alongside positions.
impl BucketRecord for u64 {
    const SIZE: usize = 8;

    #[inline]
    fn write_to(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), 8);
        out.copy_from_slice(&self.to_le_bytes());
    }

    #[inline]
    fn read_from(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 8);
        u64::from_le_bytes(bytes.try_into().unwrap())
    }
}

/// A `(position, lcp)` pair — the workhorse record for the Phase 2b
/// sample-sort partitioned merge, where each disk-spilled run carries
/// an LCP value alongside each position. Generic over the index width
/// so genome-scale inputs that fit in `u32` (anything below ~4 GB) can
/// use the 8-byte record, halving phase-1 spill / phase-4 partition
/// bytes.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SaLcp<I> {
    pub pos: I,
    pub lcp: I,
}

impl BucketRecord for SaLcp<u32> {
    const SIZE: usize = 8;

    #[inline]
    fn write_to(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), 8);
        out[0..4].copy_from_slice(&self.pos.to_le_bytes());
        out[4..8].copy_from_slice(&self.lcp.to_le_bytes());
    }

    #[inline]
    fn read_from(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 8);
        let pos = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let lcp = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        SaLcp { pos, lcp }
    }
}

impl BucketRecord for SaLcp<u64> {
    const SIZE: usize = 16;

    #[inline]
    fn write_to(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), 16);
        out[0..8].copy_from_slice(&self.pos.to_le_bytes());
        out[8..16].copy_from_slice(&self.lcp.to_le_bytes());
    }

    #[inline]
    fn read_from(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 16);
        let pos = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let lcp = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        SaLcp { pos, lcp }
    }
}

/// In-memory buffer capacity (in records). Upstream CaPS-SA uses 32 KB
/// per bucket; for 16-byte records that's 2048 entries. We use the same.
const DEFAULT_BUFFER_RECORDS: usize = 2048;

/// Common interface for the disk-backed [`ExtMemBucket`] and the
/// pure-RAM [`InMemBucket`]. Lets the ext-mem sample-sort algorithm
/// run with either storage strategy without source duplication.
pub trait BucketStore<T>: Send {
    fn add_slice(&mut self, rs: &[T]) -> io::Result<()>;
    fn mark_boundary(&mut self);
    fn total_records(&self) -> usize;
    fn boundaries(&self) -> &[usize];
    fn load_all(&mut self) -> io::Result<Vec<T>>;
}

/// Pure-RAM analogue of [`ExtMemBucket`] for the in-memory sample-sort
/// path. Records accumulate in a `Vec<T>`; `load_all` takes the vector
/// (leaving the bucket empty) — same API shape as the disk-backed
/// bucket so the sample-sort phases work against either via the
/// [`BucketStore`] trait.
pub struct InMemBucket<T> {
    records: Vec<T>,
    boundaries: Vec<usize>,
}

impl<T: Copy> InMemBucket<T> {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            boundaries: vec![0],
        }
    }
}

impl<T: Copy> Default for InMemBucket<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Send + Sync> BucketStore<T> for InMemBucket<T> {
    fn add_slice(&mut self, rs: &[T]) -> io::Result<()> {
        self.records.extend_from_slice(rs);
        Ok(())
    }

    fn mark_boundary(&mut self) {
        let last = *self.boundaries.last().unwrap();
        let now = self.records.len();
        if now != last {
            self.boundaries.push(now);
        }
    }

    fn total_records(&self) -> usize {
        self.records.len()
    }

    fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    fn load_all(&mut self) -> io::Result<Vec<T>> {
        Ok(std::mem::take(&mut self.records))
    }
}

impl<T: BucketRecord> BucketStore<T> for ExtMemBucket<T> {
    fn add_slice(&mut self, rs: &[T]) -> io::Result<()> {
        ExtMemBucket::add_slice(self, rs)
    }

    fn mark_boundary(&mut self) {
        ExtMemBucket::mark_boundary(self)
    }

    fn total_records(&self) -> usize {
        ExtMemBucket::total_records(self)
    }

    fn boundaries(&self) -> &[usize] {
        ExtMemBucket::boundaries(self)
    }

    fn load_all(&mut self) -> io::Result<Vec<T>> {
        ExtMemBucket::load_all(self)
    }
}

/// Disk-spilling sequence of `T` records.
///
/// Newly added records first go to an in-memory buffer; when the buffer
/// reaches `buffer_records` records, it is flushed to a temporary file.
/// `total_records()` reports the full logical length (in-memory + on-disk).
/// Sub-subarray boundaries can be marked by calling [`Self::mark_boundary`].
pub struct ExtMemBucket<T: BucketRecord> {
    buf: Vec<T>,
    buffer_records: usize,
    /// Lazily-created backing file. Some only after the first flush.
    file: Option<NamedTempFile>,
    /// Buffered writer over `file`. Always paired with `file`.
    writer: Option<BufWriter<File>>,
    /// Number of records already flushed to disk.
    on_disk: usize,
    /// Cumulative record count at each "boundary" — `boundaries[i]` is the
    /// total record count after the i-th sub-subarray was appended.
    /// `boundaries[0]` is always 0; the final boundary equals
    /// `total_records()`.
    boundaries: Vec<usize>,
    /// Working directory for the temp file.
    work_dir: std::path::PathBuf,
    /// Stable name prefix for debugging.
    prefix: String,
}

impl<T: BucketRecord> ExtMemBucket<T> {
    /// Create a new bucket; the backing file is created lazily on first
    /// flush. `work_dir` is the directory used for the temp file.
    ///
    /// Now only used by the unit tests — `build_ext_mem` switched to
    /// [`PooledExtMemBucket`] for its 2·p backing storage. Retained
    /// for callers who explicitly want one file per bucket.
    #[allow(dead_code)]
    pub fn new(work_dir: impl AsRef<Path>, prefix: impl Into<String>) -> Self {
        Self::with_buffer_records(work_dir, prefix, DEFAULT_BUFFER_RECORDS)
    }

    /// Like [`Self::new`] but allows a custom in-memory buffer capacity.
    #[allow(dead_code)]
    pub fn with_buffer_records(
        work_dir: impl AsRef<Path>,
        prefix: impl Into<String>,
        buffer_records: usize,
    ) -> Self {
        Self {
            buf: Vec::with_capacity(buffer_records),
            buffer_records,
            file: None,
            writer: None,
            on_disk: 0,
            boundaries: vec![0],
            work_dir: work_dir.as_ref().to_path_buf(),
            prefix: prefix.into(),
        }
    }

    /// Append a single record. Triggers a flush when the in-memory buffer
    /// reaches capacity.
    ///
    /// Currently used only by tests; the Phase 2 v1 sort+spill path uses
    /// [`Self::add_slice`] to emit an entire sorted subarray at once.
    #[allow(dead_code)]
    pub fn add(&mut self, r: T) -> io::Result<()> {
        self.buf.push(r);
        if self.buf.len() >= self.buffer_records {
            self.flush()?;
        }
        Ok(())
    }

    /// Bulk append from a slice. More efficient than repeated single
    /// [`Self::add`] when the source is already in a contiguous buffer.
    pub fn add_slice(&mut self, rs: &[T]) -> io::Result<()> {
        if self.buf.len() + rs.len() <= self.buffer_records {
            self.buf.extend_from_slice(rs);
            return Ok(());
        }
        // Spill the existing buffer first so we don't unbalance the
        // in-memory residue; then write the bulk slice directly to disk
        // and leave the buffer empty.
        self.flush()?;
        self.ensure_file()?;
        let writer = self.writer.as_mut().unwrap();
        write_records(writer, rs)?;
        self.on_disk += rs.len();
        Ok(())
    }

    /// Mark the boundary between two sub-subarrays. Called after appending
    /// one sub-subarray's worth of records; the next records start a new
    /// sub-subarray.
    ///
    /// Currently used only by tests; the Phase 2b sample-sort partitioning
    /// will mark a boundary after each subarray's contribution to a
    /// partition.
    #[allow(dead_code)]
    pub fn mark_boundary(&mut self) {
        let last = *self.boundaries.last().unwrap();
        let now = self.total_records();
        if now != last {
            self.boundaries.push(now);
        }
        // Empty contributions don't get their own boundary entry — they
        // simply don't advance `boundaries`. Callers can detect a no-op
        // by comparing two consecutive boundary entries.
    }

    /// Total number of records ever added (in-memory + on-disk).
    pub fn total_records(&self) -> usize {
        self.on_disk + self.buf.len()
    }

    /// Sub-subarray boundary cumulative counts. The i-th sub-subarray
    /// occupies records `[boundaries[i], boundaries[i+1])`. Always at
    /// least one entry (the initial 0); after `k` non-empty
    /// [`Self::mark_boundary`] calls there are `k+1` entries.
    ///
    /// Currently unused by the Phase 2 v1 streaming p-way merge; reserved
    /// for the Phase 2b sample-sort partitioned merge.
    #[allow(dead_code)]
    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    /// Flush the in-memory buffer to disk. No-op if the buffer is empty.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        self.ensure_file()?;
        let writer = self.writer.as_mut().unwrap();
        let recs = std::mem::take(&mut self.buf);
        write_records(writer, &recs)?;
        self.on_disk += recs.len();
        self.buf = Vec::with_capacity(self.buffer_records);
        Ok(())
    }

    /// Load the entire bucket contents into a freshly allocated `Vec`.
    /// After this call the in-memory buffer is empty; the on-disk file
    /// is unchanged.
    ///
    /// Currently used only by tests; the Phase 2b sample-sort partitioned
    /// merge will load each partition's bucket fully into RAM via this
    /// method.
    #[allow(dead_code)]
    pub fn load_all(&mut self) -> io::Result<Vec<T>> {
        self.flush()?;
        // The BufWriter holds bytes in user-space until flushed; the
        // reader opens a fresh OS handle and would otherwise see an
        // empty file.
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
        }
        let total = self.total_records();
        let mut out = Vec::with_capacity(total);
        if let Some(file) = self.file.as_ref() {
            let mut reader = BufReader::new(file.reopen()?);
            reader.seek(SeekFrom::Start(0))?;
            read_records(&mut reader, total, &mut out)?;
        }
        Ok(out)
    }

    /// Return a fresh `BufReader` over the bucket's contents, positioned at
    /// the start. Flushes any in-memory residue to disk first.
    ///
    /// Panics if the bucket is empty (no records were ever added) — callers
    /// should check `total_records() > 0` first.
    ///
    /// Currently unused by the Phase 2b sample-sort path (which loads each
    /// partition fully into RAM via [`Self::load_all`]); retained for the
    /// streaming p-way merge that may return as a future fast-path
    /// fallback for non-repetitive inputs.
    #[allow(dead_code)]
    pub fn open_reader(&mut self) -> io::Result<BufReader<File>> {
        self.flush()?;
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
        }
        let file = self
            .file
            .as_ref()
            .expect("open_reader on empty bucket — guard with total_records() > 0");
        let mut reader = BufReader::new(file.reopen()?);
        reader.seek(SeekFrom::Start(0))?;
        Ok(reader)
    }

    fn ensure_file(&mut self) -> io::Result<()> {
        if self.file.is_some() {
            return Ok(());
        }
        let f = tempfile::Builder::new()
            .prefix(&format!("caps-sa-{}-", self.prefix))
            .suffix(".bin")
            .tempfile_in(&self.work_dir)?;
        let writer = BufWriter::new(f.reopen()?);
        self.file = Some(f);
        self.writer = Some(writer);
        Ok(())
    }
}

/// One physical backing file shared across many [`PooledExtMemBucket`]s.
///
/// Each pooled bucket holds an `Arc<PhysicalFile>` and appends its
/// flushes at offsets handed out by `cursor.fetch_add()`. `pwrite` is
/// thread-safe at the kernel level — multiple threads can be inside
/// `write_at()` on the same fd simultaneously, since the kernel
/// serialises only by `(fd, offset_range)` and the cursor allocates
/// disjoint ranges.
struct PhysicalFile {
    file: File,
    cursor: AtomicU64,
}

/// Pool of `n_physical` anonymous-tempfile backing files.
///
/// With `p` subarray / partition buckets typically in the thousands
/// (8192 on the human genome) but only one bucket actively writing
/// per worker thread at a time, mapping `p` buckets onto
/// `num_threads` physical files cuts the openat/close/unlink syscall
/// budget from ~3·p to N. Per-bucket in-memory buffers and write
/// volumes are unchanged, so wall time is at worst neutral and on
/// networked filesystems strictly improves (NFS open is ~10 ms;
/// local open is ~50 µs).
///
/// The pool's `n_physical` argument is the *only* knob — bucket-to-
/// file assignment is the trivial `bucket_id % n_physical`. With
/// `n_physical = num_threads`, each rayon worker can flush to a
/// distinct file when bucket ids are well-distributed; the atomic
/// cursor takes care of accidental same-file collisions without a
/// lock.
pub struct BucketPool {
    files: Vec<Arc<PhysicalFile>>,
}

impl BucketPool {
    /// Open `n_physical` anonymous tempfiles in `work_dir`. They are
    /// already unlinked on creation, so no cleanup syscalls are needed
    /// on drop — the kernel reclaims them when the last fd closes.
    pub fn new(n_physical: usize, work_dir: impl AsRef<Path>) -> io::Result<Self> {
        let work_dir = work_dir.as_ref();
        let n = n_physical.max(1);
        let mut files = Vec::with_capacity(n);
        for _ in 0..n {
            let file = tempfile::tempfile_in(work_dir)?;
            files.push(Arc::new(PhysicalFile {
                file,
                cursor: AtomicU64::new(0),
            }));
        }
        Ok(Self { files })
    }

    /// Create a bucket backed by physical file `bucket_id % n_physical`.
    /// The returned bucket holds its own `Arc<PhysicalFile>`, so the
    /// file stays alive even if the pool itself drops.
    pub fn new_bucket<T: BucketRecord>(&self, bucket_id: usize) -> PooledExtMemBucket<T> {
        let phys = Arc::clone(&self.files[bucket_id % self.files.len()]);
        PooledExtMemBucket::new(phys)
    }

    /// Number of physical files in the pool.
    #[allow(dead_code)]
    pub fn n_physical(&self) -> usize {
        self.files.len()
    }
}

/// Disk-spilling bucket whose backing storage is a *region* of a
/// shared [`PhysicalFile`] rather than its own dedicated file. Bytes
/// from each flush are appended at an atomically-allocated offset and
/// recorded as a `(offset, byte_len)` extent; `load_all` reads the
/// extents back in append order to reconstruct the bucket's records.
///
/// `BucketStore<T>` impl below makes this drop-in compatible with
/// [`ExtMemBucket`] for the sample-sort phase functions.
pub struct PooledExtMemBucket<T: BucketRecord> {
    phys: Arc<PhysicalFile>,
    buf: Vec<T>,
    buffer_records: usize,
    /// `(offset, byte_len)` for each on-disk extent, in append order.
    /// `u32` byte_len bounds an extent to 4 GiB — well above any
    /// realistic single-flush size at our default
    /// `buffer_records = 2048`, but checked in debug builds.
    extents: Vec<(u64, u32)>,
    /// Cumulative record counts at each sub-subarray boundary; same
    /// semantics as [`ExtMemBucket::boundaries`].
    boundaries: Vec<usize>,
    on_disk: usize,
}

impl<T: BucketRecord> PooledExtMemBucket<T> {
    fn new(phys: Arc<PhysicalFile>) -> Self {
        Self::with_buffer_records(phys, DEFAULT_BUFFER_RECORDS)
    }

    fn with_buffer_records(phys: Arc<PhysicalFile>, buffer_records: usize) -> Self {
        Self {
            phys,
            buf: Vec::with_capacity(buffer_records),
            buffer_records,
            extents: Vec::new(),
            boundaries: vec![0],
            on_disk: 0,
        }
    }

    /// Serialise `rs` into a per-call byte buffer, allocate a disjoint
    /// region of the shared file via `cursor.fetch_add`, and `pwrite`
    /// the bytes. Records the extent so `load_all` can recover them.
    ///
    /// The scratch buffer is intentionally per-call rather than held
    /// in `self`: a persistent per-bucket scratch grows to the
    /// largest write the bucket ever sees (e.g. the full sorted
    /// subarray during phase 1 — ~6 MB at human-genome scale). With
    /// 2·p buckets each holding 6 MB, peak RSS balloons by tens of
    /// GB. Per-call allocation is ~µs each; the ~100 K flushes of a
    /// human-scale run cost ~100 ms of allocator work — well below
    /// the noise floor of the build, and worth it to keep peak RAM
    /// bounded.
    fn write_extent(&mut self, rs: &[T]) -> io::Result<()> {
        let n_records = rs.len();
        if n_records == 0 {
            return Ok(());
        }
        let n_bytes = n_records * T::SIZE;
        debug_assert!(
            n_bytes <= u32::MAX as usize,
            "single extent exceeds 4 GiB (n_bytes={n_bytes})",
        );
        let mut scratch = vec![0u8; n_bytes];
        for (i, r) in rs.iter().enumerate() {
            r.write_to(&mut scratch[i * T::SIZE..(i + 1) * T::SIZE]);
        }
        let offset = self
            .phys
            .cursor
            .fetch_add(n_bytes as u64, Ordering::Relaxed);
        pwrite_all(&self.phys.file, &scratch, offset)?;
        self.extents.push((offset, n_bytes as u32));
        self.on_disk += n_records;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        // Take the records out so write_extent doesn't need to borrow
        // `self.buf` and `self.phys` simultaneously.
        let buf = std::mem::take(&mut self.buf);
        self.write_extent(&buf)?;
        // Reuse the same `Vec<T>` — its capacity is intact, only the
        // record contents drop with the local `buf` going out of scope.
        self.buf = buf;
        self.buf.clear();
        Ok(())
    }

    /// Total number of records ever added (in-memory + on-disk).
    pub fn total_records(&self) -> usize {
        self.on_disk + self.buf.len()
    }
}

impl<T: BucketRecord> BucketStore<T> for PooledExtMemBucket<T> {
    fn add_slice(&mut self, rs: &[T]) -> io::Result<()> {
        // Pure-buffer fast path mirrors ExtMemBucket: if the new slice
        // fits in the in-memory buffer, just append.
        if self.buf.len() + rs.len() <= self.buffer_records {
            self.buf.extend_from_slice(rs);
            return Ok(());
        }
        // Otherwise flush the residue and write `rs` directly to disk.
        // We don't combine with the buffer — the two writes go into
        // separate extents but at adjacent cursor offsets, so the
        // total disk layout matches the combined-buffer variant
        // bit-for-bit and adding one more entry to `extents` is cheap.
        self.flush()?;
        self.write_extent(rs)
    }

    fn mark_boundary(&mut self) {
        let last = *self.boundaries.last().unwrap();
        let now = self.total_records();
        if now != last {
            self.boundaries.push(now);
        }
    }

    fn total_records(&self) -> usize {
        PooledExtMemBucket::total_records(self)
    }

    fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    fn load_all(&mut self) -> io::Result<Vec<T>> {
        self.flush()?;
        let total = self.total_records();
        let mut out = Vec::with_capacity(total);
        // Scratch is per-call, reused across extents within this
        // `load_all` to amortise allocation. The buffer drops here
        // when `out` is returned, so it doesn't pin RAM across the
        // many partitions phase 4 processes in parallel.
        let mut scratch: Vec<u8> = Vec::new();
        for &(offset, byte_len) in &self.extents {
            let byte_len = byte_len as usize;
            let n_records = byte_len / T::SIZE;
            if scratch.len() < byte_len {
                scratch.resize(byte_len, 0);
            }
            pread_all(&self.phys.file, &mut scratch[..byte_len], offset)?;
            for i in 0..n_records {
                out.push(T::read_from(&scratch[i * T::SIZE..(i + 1) * T::SIZE]));
            }
        }
        Ok(out)
    }
}

/// `pwrite` until all of `buf` is written. Retries on `Interrupted`.
#[cfg(unix)]
fn pwrite_all(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match file.write_at(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "pwrite returned 0",
                ));
            }
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// `pread` until all of `buf` is filled. Errors on EOF before the
/// requested length.
#[cfg(unix)]
fn pread_all(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match file.read_at(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "pread hit EOF before requested length",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

// Non-Unix targets keep the per-bucket-file path; the pooled path is
// gated on `cfg(unix)` and the workspace bench targets are Linux /
// macOS. If Windows support is needed, swap pread/pwrite for
// `seek_read`/`seek_write` from `std::os::windows::fs::FileExt`.
#[cfg(not(unix))]
compile_error!(
    "PooledExtMemBucket currently requires Unix file extension API; \
     add Windows support via seek_read/seek_write if needed."
);

fn write_records<T: BucketRecord, W: Write>(w: &mut W, rs: &[T]) -> io::Result<()> {
    // Buffer one chunk at a time to amortize allocation while keeping
    // memory bounded regardless of `rs.len()`.
    const CHUNK_RECORDS: usize = 1024;
    let mut scratch = vec![0u8; CHUNK_RECORDS * T::SIZE];
    for chunk in rs.chunks(CHUNK_RECORDS) {
        let bytes = chunk.len() * T::SIZE;
        for (i, r) in chunk.iter().enumerate() {
            r.write_to(&mut scratch[i * T::SIZE..(i + 1) * T::SIZE]);
        }
        w.write_all(&scratch[..bytes])?;
    }
    Ok(())
}

#[allow(dead_code)]
fn read_records<T: BucketRecord, R: Read>(
    r: &mut R,
    count: usize,
    out: &mut Vec<T>,
) -> io::Result<()> {
    const CHUNK_RECORDS: usize = 1024;
    let mut scratch = vec![0u8; CHUNK_RECORDS * T::SIZE];
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(CHUNK_RECORDS);
        let bytes = take * T::SIZE;
        r.read_exact(&mut scratch[..bytes])?;
        for i in 0..take {
            out.push(T::read_from(&scratch[i * T::SIZE..(i + 1) * T::SIZE]));
        }
        remaining -= take;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip_below_buffer() {
        let dir = tempdir().unwrap();
        let mut b: ExtMemBucket<SaLcp<u64>> = ExtMemBucket::new(dir.path(), "test");
        for i in 0..10 {
            b.add(SaLcp { pos: i, lcp: i * 2 }).unwrap();
        }
        assert_eq!(b.total_records(), 10);
        let loaded = b.load_all().unwrap();
        assert_eq!(loaded.len(), 10);
        for (i, r) in loaded.iter().enumerate() {
            assert_eq!(
                *r,
                SaLcp {
                    pos: i as u64,
                    lcp: (i * 2) as u64
                }
            );
        }
    }

    #[test]
    fn round_trip_with_spill() {
        let dir = tempdir().unwrap();
        // Buffer capacity 3 → spill happens on the 4th, 7th, 10th add.
        let mut b: ExtMemBucket<SaLcp<u64>> =
            ExtMemBucket::with_buffer_records(dir.path(), "spill", 3);
        for i in 0..10 {
            b.add(SaLcp { pos: i, lcp: 0 }).unwrap();
        }
        assert_eq!(b.total_records(), 10);
        let loaded = b.load_all().unwrap();
        assert_eq!(
            loaded.iter().map(|r| r.pos).collect::<Vec<_>>(),
            (0..10u64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn add_slice_bulk_path() {
        let dir = tempdir().unwrap();
        let mut b: ExtMemBucket<SaLcp<u64>> =
            ExtMemBucket::with_buffer_records(dir.path(), "bulk", 4);
        // Bulk insert larger than buffer → should hit the disk fast path.
        let mut input: Vec<SaLcp<u64>> = (0..100).map(|i| SaLcp { pos: i, lcp: 0 }).collect();
        b.add_slice(&input).unwrap();
        assert_eq!(b.total_records(), 100);
        // Add a few singles to populate the buffer afterwards.
        for i in 100..103 {
            b.add(SaLcp { pos: i, lcp: 0 }).unwrap();
        }
        input.extend((100..103).map(|i| SaLcp { pos: i, lcp: 0 }));
        let loaded = b.load_all().unwrap();
        assert_eq!(loaded, input);
    }

    #[test]
    fn boundaries_track_sub_subarrays() {
        let dir = tempdir().unwrap();
        let mut b: ExtMemBucket<SaLcp<u64>> = ExtMemBucket::new(dir.path(), "bounds");
        for i in 0..3 {
            b.add(SaLcp { pos: i, lcp: 0 }).unwrap();
        }
        b.mark_boundary();
        for i in 3..7 {
            b.add(SaLcp { pos: i, lcp: 0 }).unwrap();
        }
        b.mark_boundary();
        // Empty contribution — no new boundary entry.
        b.mark_boundary();
        for i in 7..10 {
            b.add(SaLcp { pos: i, lcp: 0 }).unwrap();
        }
        b.mark_boundary();

        assert_eq!(b.boundaries(), &[0, 3, 7, 10]);
        let loaded = b.load_all().unwrap();
        assert_eq!(loaded.len(), 10);
    }

    #[test]
    fn empty_bucket() {
        let dir = tempdir().unwrap();
        let mut b: ExtMemBucket<SaLcp<u64>> = ExtMemBucket::new(dir.path(), "empty");
        assert_eq!(b.total_records(), 0);
        let loaded = b.load_all().unwrap();
        assert!(loaded.is_empty());
    }

    /// Mirror of `round_trip_with_spill` against the pooled bucket: the
    /// in-RAM buffer fills, several extents land in the shared physical
    /// file, and `load_all` reconstructs the records in insertion order.
    #[test]
    fn pooled_round_trip_below_and_above_buffer() {
        let dir = tempdir().unwrap();
        let pool = BucketPool::new(1, dir.path()).unwrap();
        // Buffer capacity 3 → extents created on the 4th, 7th, 10th add.
        let mut b: PooledExtMemBucket<SaLcp<u64>> =
            PooledExtMemBucket::with_buffer_records(Arc::clone(&pool.files[0]), 3);
        for i in 0..10 {
            // Drive the slow path through `add_slice` so we exercise
            // both the buffer fast-path and the overflow extent path.
            b.add_slice(&[SaLcp { pos: i, lcp: 0 }]).unwrap();
        }
        assert_eq!(b.total_records(), 10);
        let loaded = b.load_all().unwrap();
        assert_eq!(
            loaded.iter().map(|r| r.pos).collect::<Vec<_>>(),
            (0..10u64).collect::<Vec<_>>()
        );
    }

    /// Two buckets sharing one physical file must not overwrite each
    /// other's bytes — the atomic cursor allocates disjoint regions.
    #[test]
    fn pooled_two_buckets_share_one_file() {
        let dir = tempdir().unwrap();
        let pool = BucketPool::new(1, dir.path()).unwrap();
        // Bucket A: pos = 0, 1, 2, …, 9.
        let mut a: PooledExtMemBucket<SaLcp<u64>> = pool.new_bucket(0);
        // Bucket B: pos = 100, 101, …, 109. `bucket_id = 1` mod 1 == 0,
        // so B lands on the same physical file as A — exactly the case
        // we need to verify.
        let mut b: PooledExtMemBucket<SaLcp<u64>> = pool.new_bucket(1);
        for i in 0..10u64 {
            a.add_slice(&[SaLcp { pos: i, lcp: 0 }]).unwrap();
            b.add_slice(&[SaLcp {
                pos: 100 + i,
                lcp: 0,
            }])
            .unwrap();
        }
        let loaded_a = a.load_all().unwrap();
        let loaded_b = b.load_all().unwrap();
        assert_eq!(
            loaded_a.iter().map(|r| r.pos).collect::<Vec<_>>(),
            (0..10u64).collect::<Vec<_>>()
        );
        assert_eq!(
            loaded_b.iter().map(|r| r.pos).collect::<Vec<_>>(),
            (100..110u64).collect::<Vec<_>>()
        );
    }

    /// Sub-subarray boundary tracking must work for the pooled bucket
    /// just as it does for the per-file [`ExtMemBucket`].
    #[test]
    fn pooled_boundaries_track_sub_subarrays() {
        let dir = tempdir().unwrap();
        let pool = BucketPool::new(2, dir.path()).unwrap();
        let mut b: PooledExtMemBucket<SaLcp<u64>> = pool.new_bucket(0);
        for chunk in &[0..3, 3..7, 7..10] {
            let records: Vec<_> = chunk
                .clone()
                .map(|i| SaLcp {
                    pos: i as u64,
                    lcp: 0,
                })
                .collect();
            b.add_slice(&records).unwrap();
            b.mark_boundary();
        }
        assert_eq!(b.boundaries(), &[0, 3, 7, 10]);
        let loaded = b.load_all().unwrap();
        assert_eq!(loaded.len(), 10);
    }

    #[test]
    fn pooled_empty_bucket() {
        let dir = tempdir().unwrap();
        let pool = BucketPool::new(2, dir.path()).unwrap();
        let mut b: PooledExtMemBucket<SaLcp<u64>> = pool.new_bucket(0);
        assert_eq!(b.total_records(), 0);
        let loaded = b.load_all().unwrap();
        assert!(loaded.is_empty());
    }
}
