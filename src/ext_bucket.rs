//! Disk-spilling bucket (Phase 2).
//!
//! Port of `Ext_Mem_Bucket.hpp` from upstream CaPS-SA. Phase 1 of the port
//! does not use external memory; this module is a placeholder so the file
//! layout matches the plan and the type/API surface can grow into it without
//! a later restructuring pass.

#![allow(dead_code)]

/// Marker. Phase 2 will replace this with the real disk-spilling bucket.
pub(crate) struct ExtMemBucket;
