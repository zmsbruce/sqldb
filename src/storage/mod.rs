use std::ops::RangeBounds;

use crate::Result;

mod disk;
mod memory;
mod mvcc;

pub trait Storage {
    type Iterator<'a>: DoubleEndedIterator<Item = Result<(Vec<u8>, Vec<u8>)>>
    where
        Self: 'a;

    /// 获取指定 key 对应的 value
    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// 将 key-value 存入数据库
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()>;

    /// 删除指定 key 对应的 value
    fn delete(&mut self, key: &[u8]) -> Result<()>;

    /// 返回一个迭代器，遍历指定范围内的 key-value
    ///
    /// # 注意
    /// 迭代器存活期间，禁止对存储进行写入或删除操作。
    fn scan<R>(&mut self, range: R) -> Self::Iterator<'_>
    where
        R: RangeBounds<Vec<u8>>;

    /// 返回一个迭代器，遍历指定前缀的 key-value
    ///
    /// # 注意
    /// 迭代器存活期间，禁止对存储进行写入或删除操作。
    fn scan_prefix(&mut self, prefix: &[u8]) -> Self::Iterator<'_> {
        let start = prefix.to_vec();
        let mut end = prefix.to_vec();
        // 需要将 end 的最后一个字节加 1，构造一个区间满足前缀要求
        // 比如 prefix 为 "abc"，则 start 为 "abc"，end 为 "abd"
        // 这样构造的区间包含了所有以 "abc" 为前缀的 key
        if let Some(last) = end.last_mut() {
            *last += 1;
        }
        self.scan(start..end) // 开区间
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use disk::DiskStorage;
    use memory::MemoryStorage;
    use tempfile::NamedTempFile;

    fn test_storage<S: Storage>(mut storage: S) {
        storage.put(b"key1", b"value1").unwrap();
        storage.put(b"key2", b"value2").unwrap();
        storage.put(b"key3", b"value3").unwrap();

        assert_eq!(storage.get(b"key1").unwrap().unwrap(), b"value1");
        assert_eq!(storage.get(b"key2").unwrap().unwrap(), b"value2");
        assert_eq!(storage.get(b"key3").unwrap().unwrap(), b"value3");

        let mut iter = storage.scan(b"key1".to_vec()..=b"key3".to_vec());
        assert_eq!(
            iter.next().unwrap().unwrap(),
            (b"key1".to_vec(), b"value1".to_vec())
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            (b"key2".to_vec(), b"value2".to_vec())
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            (b"key3".to_vec(), b"value3".to_vec())
        );
        assert!(iter.next().is_none());
        drop(iter);

        iter = storage.scan_prefix(b"key");
        assert_eq!(
            iter.next_back().unwrap().unwrap(),
            (b"key3".to_vec(), b"value3".to_vec())
        );
        assert_eq!(
            iter.next_back().unwrap().unwrap(),
            (b"key2".to_vec(), b"value2".to_vec())
        );
        assert_eq!(
            iter.next_back().unwrap().unwrap(),
            (b"key1".to_vec(), b"value1".to_vec())
        );
        drop(iter);

        storage.delete(b"key2").unwrap();
        assert_eq!(storage.get(b"key2").unwrap(), None);
    }

    #[test]
    fn test_memory_storage() {
        test_storage(MemoryStorage::new());
    }

    #[test]
    fn test_disk_storage() {
        let temp_file = NamedTempFile::new().unwrap();
        test_storage(DiskStorage::new(temp_file.path()).unwrap());
    }
}
