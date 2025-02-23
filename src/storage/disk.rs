use fs4::fs_std::FileExt;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    vec,
};

use super::Storage;
use crate::Result;

/// 基于 Bitcast 的磁盘存储，参考论文 [Bitcask: A Log-Structured Hash Table for Key/Value Data](https://riak.com/assets/bitcask-intro.pdf)。
///
/// 本实现的 `DiskStorage` 与论文中的 `Bitcask` 有一些不同：
/// - 论文中的 `Bitcask` 使用了 `merge` 操作，将多个日志文件合并为一个新的日志文件，以减少磁盘空间的使用；
/// - 本实现的 `DiskStorage` 没有实现 `merge` 操作，而是提供一个方法，将数据重新写入一个新的日志文件，从而去除已经删除的数据。
///
/// # 存储结构
///
/// - **磁盘上的 Log**：| key_len | val_len | key | value |；
/// - **内存中的 KeyDir**：`BTreeMap`，key 为 key，value 为 (磁盘上 value 的索引, value 的长度)；
///
/// # 读取数据
/// 1. 根据 key 在内存中的 KeyDir 中查找 key 对应的 (offset, len)；
/// 2. 根据 offset 读取磁盘上的数据；
///
/// # 写入数据
/// 1. 将 key 和 value 写入磁盘，并获取 offset 和 value 的长度；
/// 2. 将 key 和 (offset, len) 写入内存中的 KeyDir；
///
/// # 删除数据
/// 1. 将 value 的长度设置为 0，然后将 key 和 value 写入磁盘，并获取 offset 和 value 的长度；
/// 2. 将 key 和 (offset, len) 写入内存中的 KeyDir；
pub struct DiskStorage {
    keydir: BTreeMap<Vec<u8>, (u64, u64)>,
    log: File,
    log_path: PathBuf,
}

impl DiskStorage {
    /// 创建一个新的 `DiskStorage` 实例。
    pub fn new<P>(filename: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let file_path = filename.as_ref().to_path_buf();
        // 如果文件所在的目录不存在，则创建目录
        // 如果目录为 None，则会在创建文件时报错
        if let Some(dir) = file_path.parent() {
            if !dir.exists() {
                std::fs::create_dir_all(dir)?;
            }
        }

        // 打开文件，如果文件不存在则创建文件，设置为追加写入
        let file = fs::OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&file_path)?;
        // 尝试获取文件的独占锁
        file.try_lock_exclusive()?;

        // 如果存在压缩后的文件，则删除压缩后的文件
        if file_path.with_extension("compact").exists() {
            fs::remove_file(file_path.with_extension("compact"))?;
        }

        let keydir = BTreeMap::new();

        let mut storage = DiskStorage {
            keydir,
            log: file,
            log_path: file_path,
        };
        storage.build_keydir()?; // 从磁盘上读取数据，构建 KeyDir

        Ok(storage)
    }

    /// 从磁盘中读取数据，构建 KeyDir。
    fn build_keydir(&mut self) -> Result<()> {
        let file_sz = self.log.metadata()?.len();
        let mut file_reader = BufReader::new(&self.log);

        loop {
            let offset = file_reader.stream_position()?;
            if offset >= file_sz {
                break;
            }
            let mut len_buf = [0u8; u64::BITS as usize / 8];

            // 读取 key_len
            file_reader.read_exact(&mut len_buf)?;
            let key_len = u64::from_le_bytes(len_buf);

            // 读取 val_len
            file_reader.read_exact(&mut len_buf)?;
            let val_len = u64::from_le_bytes(len_buf);

            // 读取 key
            let mut key = vec![0u8; key_len as usize];
            file_reader.read_exact(&mut key)?;

            if (val_len >> (u64::BITS - 1)) != 0 {
                // 如果 val_len 最高位为 1，则表示该 key 对应的数据已经被删除
                self.keydir.remove(&key);
                // 跳过 value
                file_reader.seek(SeekFrom::Current(
                    // val_len & !(1 << (u64::BITS - 1)) 表示去除最高位的 val_len
                    (val_len & !(1 << (u64::BITS - 1))) as i64,
                ))?;
            } else {
                // 将 key 和 (offset, len) 写入 KeyDir
                self.keydir
                    .insert(key, (offset + u64::BITS as u64 / 8 * 2 + key_len, val_len));
                // 跳过 value
                file_reader.seek(SeekFrom::Current(val_len as i64))?;
            }
        }
        Ok(())
    }

    /// 压缩日志文件
    ///
    /// 将日志文件中的数据重新写入一个新的文件中，然后将新文件重命名为原文件，从而去除已经删除的数据。
    pub fn compact(&mut self) -> Result<()> {
        // 创建一个新的日志文件
        let new_log_path = self.log_path.with_extension("compact");

        let mut new_log = fs::OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&new_log_path)?;

        // 将数据重新写入新的日志文件
        // 由于 KeyDir 中没有删除的数据，所以只需要将 KeyDir 中的数据重新写入新的日志文件即可
        for (key, (offset, len)) in self.keydir.iter() {
            // 读取 value 对应的数据
            self.log.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            self.log.read_exact(&mut buf)?;

            // 将 key 和 value 写入新的日志文件
            new_log.seek(SeekFrom::End(0))?;
            let total_len = u64::BITS as usize / 8 * 2 + key.len() + buf.len();
            let mut writer = BufWriter::with_capacity(total_len, &new_log);
            writer.write_all(&(key.len() as u64).to_le_bytes())?;
            writer.write_all(&(buf.len() as u64).to_le_bytes())?;
            writer.write_all(key)?;
            writer.write_all(&buf)?;
            writer.flush()?;
        }

        // 重命名新日志文件为原日志文件
        fs::rename(&new_log_path, &self.log_path)?;

        // 重新打开文件进行读取操作
        self.log = fs::OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.log_path)?;
        self.log.try_lock_exclusive()?;

        Ok(())
    }
}

pub struct DiskStorageIterator<'a> {
    inner: std::collections::btree_map::Range<'a, Vec<u8>, (u64, u64)>,
    file: &'a mut File,
}

impl Iterator for DiskStorageIterator<'_> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, (offset, len))| {
            self.file.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            self.file.read_exact(&mut buf)?;

            Ok((k.clone(), buf))
        })
    }
}

impl DoubleEndedIterator for DiskStorageIterator<'_> {
    fn next_back(&mut self) -> Option<Result<(Vec<u8>, Vec<u8>)>> {
        self.inner.next_back().map(|(k, (offset, len))| {
            self.file.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            self.file.read_exact(&mut buf)?;

            Ok((k.clone(), buf))
        })
    }
}

impl Storage for DiskStorage {
    type Iterator<'a> = DiskStorageIterator<'a>;

    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let offset = self.log.seek(SeekFrom::End(0))?;

        let mut writer = BufWriter::with_capacity(
            usize::BITS as usize / 8 * 2 + key.len() + value.len(),
            &self.log,
        );
        writer.write_all(&(key.len() as u64).to_le_bytes())?;
        writer.write_all(&(value.len() as u64).to_le_bytes())?;
        writer.write_all(key)?;
        writer.write_all(value)?;
        writer.flush()?;

        self.keydir.insert(
            key.to_vec(),
            (
                offset + usize::BITS as u64 / 8 * 2 + key.len() as u64,
                value.len() as u64,
            ),
        );

        Ok(())
    }

    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some((offset, len)) = self.keydir.get(key) {
            self.log.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            self.log.read_exact(&mut buf)?;

            Ok(Some(buf))
        } else {
            Ok(None)
        }
    }

    fn delete(&mut self, key: &[u8]) -> Result<()> {
        if let Some((_, val_len)) = self.keydir.get(key) {
            self.log.seek(SeekFrom::End(0))?;
            let total_len = u64::BITS as usize / 8 * 2 + key.len();
            let mut writer = BufWriter::with_capacity(total_len, &self.log);
            writer.write_all(&(key.len() as u64).to_le_bytes())?;
            writer.write_all(&(val_len | (1 << (u64::BITS - 1))).to_le_bytes())?;
            writer.write_all(key)?;

            self.keydir.remove(key);
        }
        Ok(())
    }

    fn scan<R>(&mut self, range: R) -> Self::Iterator<'_>
    where
        R: std::ops::RangeBounds<Vec<u8>>,
    {
        let inner = self.keydir.range(range);
        DiskStorageIterator {
            inner,
            file: &mut self.log,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_build_keydir() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key1value1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key2value2\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key3value3\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x80key2value2").unwrap();

        let storage = DiskStorage::new(file.path()).unwrap();
        assert_eq!(storage.keydir.len(), 2);
        assert_eq!(*storage.keydir.get(&b"key1"[..]).unwrap(), (20, 6));
        assert!(!storage.keydir.contains_key(&b"key2"[..]));
        assert_eq!(*storage.keydir.get(&b"key3"[..]).unwrap(), (72, 6));
    }

    #[test]
    fn test_compact() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key1value1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key2value2\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key3value3\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x80key2value2").unwrap();

        let mut storage = DiskStorage::new(file.path()).unwrap();
        storage.compact().unwrap();

        let mut buf = Vec::new();
        storage.log.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key1value1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key3value3");
    }
}
