use rustc_hash::{FxHashMap, FxHashSet};

pub type HashMap<K, V> = FxHashMap<K, V>;

#[allow(dead_code)]
pub type HashSet<K> = FxHashSet<K>;
