#!/usr/bin/env python3
"""Build an annotation-shaped splice-junction fixture for `gsj_bench`.

    bench/gsj_fixture.py <genome.fa> <annotation.gtf> <out-prefix> [overhang]

Writes `<prefix>.text` (one byte per symbol, A/C/G/T/N as 0..=4) and
`<prefix>.seg` (packed little-endian u64 segment lengths).

The layout mirrors what a STAR-style index constructs: the genome sequence,
then one 2*overhang flank per distinct splice junction, then the reverse
complement of the whole thing. Each junction flank is its own segment, which
is what makes the comparator segmented; the genome is one segment per record.

Note this is a *shape* reproduction. A genome-wide annotation yields far more
junctions than a single-chromosome one, so segment counts differ accordingly.
"""
import collections
import struct
import sys

CODE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
COMP = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}


def main() -> None:
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    fasta, gtf, prefix = sys.argv[1:4]
    overhang = int(sys.argv[4]) if len(sys.argv) == 5 else 100

    seq = bytearray()
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            for ch in line.strip().upper():
                seq.append(CODE.get(ch, 4))

    transcripts = collections.defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            i = f[8].find('transcript_id "')
            if i < 0:
                continue
            tid = f[8][i + 15 : f[8].find('"', i + 15)]
            transcripts[tid].append((int(f[3]) - 1, int(f[4])))

    junctions = set()
    for exons in transcripts.values():
        exons.sort()
        for k in range(len(exons) - 1):
            donor, acceptor = exons[k][1], exons[k + 1][0]
            if acceptor > donor:
                junctions.add((donor, acceptor))

    flanks = bytearray()
    seglens = [len(seq)]
    for donor, acceptor in sorted(junctions):
        left = seq[max(0, donor - overhang) : donor]
        right = seq[acceptor : acceptor + overhang]
        flanks += left + bytearray([4] * (overhang - len(left)))
        flanks += right + bytearray([4] * (overhang - len(right)))
        seglens.append(2 * overhang)

    forward = seq + flanks
    text = forward + bytearray(COMP[b] for b in reversed(forward))
    seglens = seglens + list(reversed(seglens))
    assert sum(seglens) == len(text)

    with open(f"{prefix}.text", "wb") as out:
        out.write(bytes(text))
    with open(f"{prefix}.seg", "wb") as out:
        out.write(struct.pack(f"<{len(seglens)}Q", *seglens))

    counts = collections.Counter(text)
    print(
        f"{prefix}: {len(text)} symbols, {len(seglens)} segments, "
        f"{len(junctions)} junctions, "
        f"{sum(v for k, v in counts.items() if k < 4)} ACGT-start positions",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
