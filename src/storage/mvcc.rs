use std::{
    collections::HashSet,
    ops::Add,
    sync::{Arc, Mutex, MutexGuard},
};

use serde::{Deserialize, Serialize};

use super::Storage;
use crate::{
    Error::{InternalError, WriteConflict},
    Result,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Version(u64);

impl Version {
    pub fn encode(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self).map_err(|e| e.into())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| e.into())
    }

    pub fn max() -> Self {
        Self(u64::MAX)
    }

    pub fn min() -> Self {
        Self(0)
    }
}

impl Add<u64> for Version {
    type Output = Self;

    fn add(self, rhs: u64) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl From<u64> for Version {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

type Key = Vec<u8>;

/// MVCC 存储引擎的 key
///
/// - `NextVersion`: 下一个版本号
/// - `TxnActive`: 活跃事务
/// - `TxnWrite`: 事务写入记录，用于回滚事务
/// - `Version`: 版本记录，用于事务的可见性判断
#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum MvccKey {
    NextVersion,
    TxnActive(Version),
    TxnWrite(Version, Key),
    Version(Key, Version),
}

impl MvccKey {
    /// 编码 key
    pub fn encode(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self).map_err(|e| e.into())
    }

    /// 解码 key
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| e.into())
    }
}

/// MVCC 存储引擎的 key 前缀，用于扫描一个范围使用
#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum MvccKeyPrefix {
    NextVersion,
    TxnActive,
    TxnWrite(Version),
    Version(Key),
}

impl MvccKeyPrefix {
    /// 编码 key 前缀
    pub fn encode(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self).map_err(|e| e.into())
    }
}

/// MVCC 存储引擎
pub struct Mvcc<S: Storage> {
    storage: Arc<Mutex<S>>,
    current_version: Version,
    active_versions: HashSet<Version>,
}

impl<S: Storage> Mvcc<S> {
    /// 开启一个新事务
    pub fn begin(s: Arc<Mutex<S>>) -> Result<Self> {
        // 获取当前存储引擎的锁
        let mut storage = s.lock()?;

        // 获取下一个版本号，如果不存在则从 1 开始
        let next_version = if let Some(value) = storage.get(&MvccKey::NextVersion.encode()?)? {
            Version::decode(&value)?
        } else {
            Version(1)
        };

        // 将下一个版本号加 1，写入存储引擎
        storage.put(
            &MvccKey::NextVersion.encode()?,
            &bincode::serialize(&(next_version + 1))?,
        )?;

        // 将新事务加入活跃事务列表
        storage.put(&MvccKey::TxnActive(next_version).encode()?, &[])?;

        // 扫描所有活跃事务
        let active_versions = Self::scan_active_txn(&mut storage)?;

        Ok(Self {
            storage: s.clone(),
            current_version: next_version,
            active_versions,
        })
    }

    /// 查找所有活跃事务
    fn scan_active_txn(storage: &mut MutexGuard<S>) -> Result<HashSet<Version>> {
        let mut active_versions = HashSet::new();

        // 扫描前缀为 TxnActive 的 key
        let mut iter = storage.scan_prefix(&MvccKeyPrefix::TxnActive.encode()?);
        while let Some((key, _)) = iter.next().transpose()? {
            // 解码 key，获取事务版本，并加入活跃事务列表
            if let MvccKey::TxnActive(version) = MvccKey::decode(&key)? {
                active_versions.insert(version);
            } else {
                return Err(InternalError(format!(
                    "unexpected key {} when scanning active transactions",
                    String::from_utf8_lossy(key.as_slice())
                )));
            }
        }
        Ok(active_versions)
    }

    /// 版本是否可见
    ///
    /// 版本可见的条件是：
    ///
    /// - 版本小于等于当前版本；
    /// - 版本不在活跃事务列表中。
    #[inline]
    fn is_version_visible(&self, version: Version) -> bool {
        version <= self.current_version && !self.active_versions.contains(&version)
    }

    /// 更新/删除数据的内置函数
    ///
    /// - 如果 `value` 为 `None`，则删除 `key` 对应的数据
    /// - 否则更新 `key` 对应的数据
    fn write_inner(&self, key: Key, value: Option<Vec<u8>>) -> Result<()> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 活跃事务和大于当前版本的事务都不可见
        // 取活跃事务的最小值到可能存在的版本最大值，构成一个范围，其中会包括所有不可见的事务
        let begin = self
            .active_versions
            .iter()
            .min()
            .copied()
            .unwrap_or(self.current_version + 1);
        let begin_key = MvccKey::Version(key.clone(), begin).encode()?;
        let end_key = MvccKey::Version(key.clone(), Version::max()).encode()?;

        // 检查是否有不可见的版本写入了 key
        // 首先根据活跃事务和大于当前版本的事务的范围，找到最后一个可能不可见的事务
        // 如果这个事务不可见，则说明有不可见的事务写入了 key，返回写冲突
        //? 为什么只需检查最后一个可能不可见的版本即可：
        //? 若最后版本不可见：直接判定存在写冲突，无需检查更早的版本，因为该版本是当前事务可能冲突的最高版本。
        //? 若最后版本可见：所有更早的版本要么已被提交（可见），要么会发生写冲突。
        if let Some((key, _)) = storage.scan(begin_key..=end_key).last().transpose()? {
            if let MvccKey::Version(_, version) = MvccKey::decode(&key)? {
                if !self.is_version_visible(version) {
                    return Err(WriteConflict);
                }
            } else {
                return Err(InternalError(format!(
                    "unexpected key {} when scanning versions",
                    String::from_utf8_lossy(key.as_slice())
                )));
            }
        }

        // 记录新版本写入了哪些 key，用于回滚事务
        storage.put(
            &MvccKey::TxnWrite(self.current_version, key.clone()).encode()?,
            &[],
        )?;

        // 如果 value 不为 None，则写入新的数据，否则删除数据
        if let Some(value) = value {
            storage.put(
                &MvccKey::Version(key, self.current_version).encode()?,
                &value,
            )?;
        } else {
            storage.delete(&MvccKey::Version(key, self.current_version).encode()?)?;
        }

        Ok(())
    }

    /// 更新 `key` 对应的值
    #[inline]
    pub fn set(&self, key: Key, value: Vec<u8>) -> Result<()> {
        self.write_inner(key, Some(value))
    }

    /// 删除 `key` 对应的值
    #[inline]
    pub fn delete(&self, key: Key) -> Result<()> {
        self.write_inner(key, None)
    }

    /// 获取 `key` 对应的值
    pub fn get(&self, key: Key) -> Result<Option<Vec<u8>>> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 设置范围为 0 到当前版本，因为大于当前版本的事务一定不可见
        let begin = MvccKey::Version(key.clone(), Version::min()).encode()?;
        let end = MvccKey::Version(key.clone(), self.current_version).encode()?;

        // 从范围中找到最新的可见版本
        let mut iter = storage.scan(begin..end).rev(); // 新版本在后面
        while let Some((key, value)) = iter.next().transpose()? {
            if let MvccKey::Version(_, version) = MvccKey::decode(&key)? {
                // 判断是否可见，此处指的是不在活跃事务中，因为范围已经排除了大于当前版本的事务
                if self.is_version_visible(version) {
                    return Ok(Some(value));
                }
            } else {
                return Err(InternalError(format!(
                    "unexpected key {} when scanning versions",
                    String::from_utf8_lossy(key.as_slice())
                )));
            }
        }

        // 没有找到可见版本，返回 None
        Ok(None)
    }

    /// 扫描 `prefix` 开头的所有可见的事务记录
    pub fn scan_visible_versions(&self, prefix: Key) -> Result<Vec<(Key, Vec<u8>)>> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        let prefix = MvccKeyPrefix::Version(prefix).encode()?;
        let result = storage
            .scan_prefix(&prefix)
            .map(|item| {
                let (key, value) = item?;
                match MvccKey::decode(&key)? {
                    // 如果版本可见，则返回 key-value，之后的过滤中被保留
                    MvccKey::Version(_, version) if self.is_version_visible(version) => {
                        Ok(Some((key, value)))
                    }
                    // 否则返回 None，之后被过滤掉
                    MvccKey::Version(_, _) => Ok(None),
                    // 如果解析不是 Version，则返回错误
                    _ => Err(InternalError(format!(
                        "unexpected key {} when scanning versions",
                        String::from_utf8_lossy(&key)
                    )))?,
                }
            })
            .filter_map(|x| x.transpose())
            .collect::<Result<Vec<_>>>()?;

        Ok(result)
    }

    /// 提交事务
    ///
    /// 对于提交事务，实际上是让这个事务的修改对后续新开启的事务是可见的。
    /// 因此，只需要将当前事务对应的所有 TxnWrite 记录，以及当前事务在活跃事务列表中的记录删除即可。
    pub fn commit(&self) -> Result<()> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 找到当前事务对应的所有 TxnWrite 记录
        let txn_keys = storage
            .scan_prefix(&MvccKeyPrefix::TxnWrite(self.current_version).encode()?)
            .map(|item| {
                let (key, _) = item?;
                if let MvccKey::TxnWrite(_, key) = MvccKey::decode(&key)? {
                    Ok(key)
                } else {
                    Err(InternalError(format!(
                        "unexpected key {} when scanning txn writes",
                        String::from_utf8_lossy(&key)
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 将当前事务对应的所有 TxnWrite 记录从存储引擎中删除
        for key in txn_keys {
            storage.delete(&key)?;
        }

        // 将当前事务从活跃事务列表中移除
        storage.delete(&MvccKey::TxnActive(self.current_version).encode()?)?;

        Ok(())
    }

    /// 回滚事务
    pub fn rollback(&self) -> Result<()> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 找到当前事务对应的所有 TxnWrite 记录，并转换为 Version 记录
        let txn_keys = storage
            .scan_prefix(&MvccKeyPrefix::TxnWrite(self.current_version).encode()?)
            .map(|item| {
                let (key, _) = item?;
                if let MvccKey::TxnWrite(_, key) = MvccKey::decode(&key)? {
                    Ok(MvccKey::Version(key, self.current_version).encode()?)
                } else {
                    Err(InternalError(format!(
                        "unexpected key {} when scanning txn writes",
                        String::from_utf8_lossy(&key)
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 将当前事务对应的所有 Version 记录从存储引擎中删除
        for key in txn_keys {
            storage.delete(&key)?;
        }

        // 将当前事务从活跃事务列表中移除
        storage.delete(&MvccKey::TxnActive(self.current_version).encode()?)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvcc_key_codec() {
        let key = MvccKey::NextVersion;
        let encoded = key.encode().unwrap();
        let decoded = MvccKey::decode(&encoded).unwrap();
        assert_eq!(key, decoded);

        let key = MvccKey::TxnActive(1.into());
        let encoded = key.encode().unwrap();
        let decoded = MvccKey::decode(&encoded).unwrap();
        assert_eq!(key, decoded);

        let key = MvccKey::TxnWrite(1.into(), b"key".to_vec());
        let encoded = key.encode().unwrap();
        let decoded = MvccKey::decode(&encoded).unwrap();
        assert_eq!(key, decoded);

        let key = MvccKey::Version(b"key".to_vec(), 1.into());
        let encoded = key.encode().unwrap();
        let decoded = MvccKey::decode(&encoded).unwrap();
        assert_eq!(key, decoded);
    }

    #[test]
    fn test_mvcckey_encode() {
        let key_1 = MvccKey::TxnActive(1.into());
        let encoded_1 = key_1.encode().unwrap();

        let key_2 = MvccKey::TxnActive(114514.into());
        let encoded_2 = key_2.encode().unwrap();

        assert_ne!(encoded_1, encoded_2);

        let key_3 = MvccKey::NextVersion;
        let encoded_3 = key_3.encode().unwrap();
        assert_ne!(encoded_1, encoded_3);

        let key_4 = MvccKey::TxnActive(114514.into());
        let encoded_4 = key_4.encode().unwrap();
        assert_eq!(encoded_2, encoded_4);
    }

    #[test]
    fn test_mvcckey_encode_prefix() {
        let key_prefix_1 = MvccKeyPrefix::TxnActive;
        let encoded_prefix_1 = key_prefix_1.encode().unwrap();

        let key_1 = MvccKey::TxnActive(114514.into());
        let encoded_1 = key_1.encode().unwrap();
        assert!(encoded_1.starts_with(&encoded_prefix_1));

        let key_prefix_2 = MvccKeyPrefix::Version(b"key".to_vec());
        let encoded_prefix_2 = key_prefix_2.encode().unwrap();

        let key_2 = MvccKey::Version(b"key".to_vec(), 114514.into());
        let encoded_2 = key_2.encode().unwrap();
        assert!(encoded_2.starts_with(&encoded_prefix_2));
        assert!(!encoded_2.starts_with(&encoded_prefix_1));
    }
}
