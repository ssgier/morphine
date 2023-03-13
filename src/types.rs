use rustc_hash::FxHashMap;

pub type HashMap<K, V> = FxHashMap<K, V>;
#[cfg(test)]
pub mod tests {
    use rustc_hash::FxHashSet;

    pub type HashSet<K> = FxHashSet<K>;
}
