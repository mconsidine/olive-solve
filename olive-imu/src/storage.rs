// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::path::PathBuf;

pub trait PersistentStorage: Send + Sync {
    fn get(&self, key: &str) -> Option<String>;
    fn set(&self, key: &str, value: &str);
    fn remove(&self, key: &str);
}

pub struct FileStorage {
    dir: PathBuf,
}

impl FileStorage {
    pub fn new(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Self { dir }
    }

    fn get_path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{}.txt", key))
    }
}

impl PersistentStorage for FileStorage {
    fn get(&self, key: &str) -> Option<String> {
        std::fs::read_to_string(self.get_path(key)).ok()
    }

    fn set(&self, key: &str, value: &str) {
        let _ = std::fs::write(self.get_path(key), value);
    }

    fn remove(&self, key: &str) {
        let _ = std::fs::remove_file(self.get_path(key));
    }
}
