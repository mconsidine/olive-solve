use chrono::{Datelike, Timelike};
use ndarray::Array2;
use olive_imu::storage::PersistentStorage;
use olive_imu::{Imu, ImuDevice, MountCoordinates};
use std::sync::Arc;
use std::time::SystemTime;
use tetra3::extractor::Extractor;
use tetra3::solver::{Solution, SolveOptions, SolveStatus, Solver};
use tokio::sync::RwLock;

/// Wrapper to inject custom IMU devices (e.g. for testing)
pub struct CustomImuWrapper(pub Box<dyn ImuDevice + Send>);

impl ImuDevice for CustomImuWrapper {
    fn init(&mut self) -> Result<(), String> {
        self.0.init()
    }
    fn poll_gyros(&mut self) -> Result<Vec<(nalgebra::Vector3<f64>, f64)>, String> {
        self.0.poll_gyros()
    }
    fn revive(&mut self) -> Result<(), String> {
        self.0.revive()
    }
}

/// Defines the type of hardware IMU to start or fallback to.
pub enum ImuType {
    None,
    Auto,
    Bno085,
    Bmi160,
    Custom(Box<dyn ImuDevice + Send + Sync>),
}

/// The source of the celestial position estimate.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionSource {
    /// Position derived directly from a successful plate solve.
    Solver,
    /// Position estimated by the IMU using the last known solver anchor.
    Imu,
    /// The solver recently failed (e.g. due to movement or obstruction),
    /// and the IMU is unavailable to provide a real-time estimate.
    /// This represents the last known good plate-solved anchor, but it is physically stale.
    SolverStale,
}

/// A unified representation of the device's celestial orientation.
#[derive(Debug, Clone)]
pub struct Position {
    /// Right Ascension in degrees.
    pub ra: f64,
    /// Declination in degrees.
    pub dec: f64,
    /// Roll in degrees.
    pub roll: f64,
    /// The source that provided this position estimate.
    pub source: PositionSource,
    /// The time at which this position was valid.
    pub timestamp: SystemTime,
}

/// The unified solver coordinating tetra3 plate solving and olive-imu hardware tracking.
/// Handles coordinate transformations between the equatorial and local horizontal frames.
pub struct FusedSolver {
    solver: Arc<RwLock<Option<Solver>>>,
    extractor: Arc<RwLock<Option<Extractor>>>,
    imu: Arc<RwLock<Option<Arc<Imu>>>>,
    imu_type: Arc<RwLock<ImuType>>,
    storage: Option<Arc<dyn PersistentStorage>>,
    latest_solve_position: Arc<RwLock<Option<Position>>>,
    last_solve_failed: Arc<RwLock<bool>>,

    // Observer location required for Alt/Az IMU coordinate mapping
    latitude: Arc<RwLock<Option<f64>>>,
    longitude: Arc<RwLock<Option<f64>>>,
}

impl FusedSolver {
    /// Initialize the unified solver.
    /// Takes in the database location, an optional IMU type (defaults to Auto),
    /// and an optional storage implementation. Location must be set later.
    pub async fn new(
        database_path: &std::path::Path,
        imu_type: Option<ImuType>,
        storage: Option<Arc<dyn PersistentStorage>>,
    ) -> Result<Self, String> {
        let solver = Solver::load_database(database_path)
            .map_err(|e| format!("Failed to load database: {:?}", e))?;
        let extractor = Extractor::new();

        Ok(Self {
            solver: Arc::new(RwLock::new(Some(solver))),
            extractor: Arc::new(RwLock::new(Some(extractor))),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(imu_type.unwrap_or(ImuType::Auto))),
            storage,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        })
    }

    /// Sets the observer location. Must be called before IMU mapping can occur.
    pub async fn set_observer_location(&self, lat: f64, lon: f64) {
        *self.latitude.write().await = Some(lat);
        *self.longitude.write().await = Some(lon);
    }

    /// Starts the IMU. Requires location to be set first. Returns true if successful.
    pub async fn start_imu(&self) -> Result<bool, String> {
        if self.latitude.read().await.is_none() || self.longitude.read().await.is_none() {
            return Err("Observer location must be set before starting the IMU.".into());
        }

        let mut imu_lock = self.imu.write().await;
        if imu_lock.is_some() {
            return Err("IMU is already running.".into());
        }

        let mut imu_type = self.imu_type.write().await;
        let imu_instance = match &mut *imu_type {
            ImuType::None => return Err("Cannot start IMU because ImuType is None.".into()),
            ImuType::Auto => {
                // Try Bno085 primary/alt
                if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4A, true) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4B, true) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x68) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x69) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("Auto mode could not find any supported IMU hardware.".into());
                }
            }
            ImuType::Bno085 => {
                if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4A, true) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4B, true) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("BNO085 hardware not found on I2C bus.".into());
                }
            }
            ImuType::Bmi160 => {
                if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x68) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x69) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("BMI160 hardware not found on I2C bus.".into());
                }
            }
            ImuType::Custom(_) => {
                let mut temp = ImuType::None;
                std::mem::swap(&mut *imu_type, &mut temp);
                if let ImuType::Custom(dev) = temp {
                    Some(Imu::start(CustomImuWrapper(dev), self.storage.clone())?)
                } else {
                    None
                }
            }
        };

        if let Some(instance) = imu_instance {
            *imu_lock = Some(Arc::new(instance));
            Ok(true)
        } else {
            Err("Failed to start IMU.".into())
        }
    }

    /// Stops the IMU and drops the background polling thread. Safe to call if not started.
    pub async fn stop_imu(&self) -> Result<(), String> {
        let mut imu_lock = self.imu.write().await;
        *imu_lock = None;
        Ok(())
    }

    /// Resets the internal IMU zero-bias and deletes the SVD calibration matrix from persistent storage.
    pub async fn reset_calibration(&self) {
        if let Some(ref imu) = *self.imu.read().await {
            imu.clear_calibration().await;
            imu.reset_bias_calibration();
        }
    }

    // ==========================================
    // TETRA3 WRAPPERS
    // ==========================================

    /// Extracts star centroids from raw image data (stub implementation).
    pub async fn extract(&self, _image_data: &[u8]) -> Result<Array2<f64>, String> {
        let _extractor_guard = self.extractor.write().await;
        // Extractor::extract is not fully wrapped here for arbitrary byte buffers.
        Ok(Array2::zeros((0, 2)))
    }

    /// Performs a plate solve using pre-extracted centroids and given image dimensions.
    /// If successful, the solver automatically updates the IMU anchor internally.
    pub async fn solve_from_centroids(
        &self,
        centroids: &Array2<f64>,
        size: (f64, f64),
        options: SolveOptions,
        timestamp: Option<SystemTime>,
    ) -> Result<Solution, String> {
        let time = timestamp.unwrap_or_else(SystemTime::now);

        let solution = {
            let mut solver_guard = self.solver.write().await;
            if let Some(solver) = solver_guard.as_mut() {
                solver.solve(centroids, size, options)
            } else {
                return Err("Solver is not initialized.".into());
            }
        };

        if solution.status == SolveStatus::MatchFound {
            *self.last_solve_failed.write().await = false;
            self.update_anchor_from_solution(&solution, time).await;
        } else {
            *self.last_solve_failed.write().await = true;
        }

        Ok(solution)
    }

    /// Extracts star centroids from the image and performs a plate solve.
    /// If successful, the solver automatically updates the IMU anchor internally.
    pub async fn solve_from_image(
        &self,
        image_data: &[u8],
        size: (f64, f64),
        options: SolveOptions,
        timestamp: Option<SystemTime>,
    ) -> Result<Solution, String> {
        let centroids_result = self.extract(image_data).await;

        match centroids_result {
            Ok(centroids) => {
                self.solve_from_centroids(&centroids, size, options, timestamp)
                    .await
            }
            Err(e) => {
                *self.last_solve_failed.write().await = true;
                Err(e)
            }
        }
    }

    async fn update_anchor_from_solution(&self, solution: &Solution, time: SystemTime) {
        let ra = if let (Some(t_ra), Some(_t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_ra[0]
        } else {
            solution.ra.unwrap_or(0.0)
        };
        let dec = if let (Some(_t_ra), Some(t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_dec[0]
        } else {
            solution.dec.unwrap_or(0.0)
        };
        let roll = solution.roll.unwrap_or(0.0);

        if let Some(ref imu) = *self.imu.read().await {
            let lat_opt = *self.latitude.read().await;
            let lon_opt = *self.longitude.read().await;

            if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                let dt: chrono::DateTime<chrono::Utc> = time.into();

                let (alt, az, alt_az_roll) = ra_dec_to_alt_az(ra, dec, roll, lat, lon, dt);

                let mount_coords = MountCoordinates {
                    pitch: alt,
                    yaw: az,
                    roll: alt_az_roll,
                };
                imu.update_anchor(&mount_coords, &time).await;
            }
        }

        *self.latest_solve_position.write().await = Some(Position {
            ra,
            dec,
            roll,
            source: PositionSource::Solver,
            timestamp: time,
        });
    }

    // ==========================================
    // IMU STATUS WRAPPERS
    // ==========================================

    /// Retrieves the real-time calibration metrics from the IMU hardware, if running.
    pub async fn get_calibration_status(&self) -> Option<olive_imu::TransformMetrics> {
        if let Some(ref imu) = *self.imu.read().await {
            imu.get_calibration_metrics().await
        } else {
            None
        }
    }

    /// Retrieves the real-time motion stability state from the IMU hardware, if running.
    pub async fn get_motion_state(&self) -> Option<olive_imu::MotionState> {
        if let Some(ref imu) = *self.imu.read().await {
            Some(imu.get_motion_state())
        } else {
            None
        }
    }

    /// Fetches the latest known orientation of the device.
    /// If the IMU is actively tracking and has a valid plate solve anchor, this returns the real-time IMU estimate.
    /// Otherwise, it safely falls back to returning the position from the last successful plate solve.
    pub async fn get_latest_position(&self) -> Option<Position> {
        let mut last_solve = self.latest_solve_position.read().await.clone();
        let last_failed = *self.last_solve_failed.read().await;

        if let Some(ref imu) = *self.imu.read().await {
            if let Ok((est, is_imu_estimate)) = imu.get_estimated_pointing(&SystemTime::now()).await
            {
                let lat_opt = *self.latitude.read().await;
                let lon_opt = *self.longitude.read().await;

                if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                    let dt_now = chrono::Utc::now();
                    let (current_ra, current_dec, current_roll) =
                        alt_az_to_ra_dec(est.pitch, est.yaw, est.roll, lat, lon, dt_now);

                    let source = if is_imu_estimate {
                        PositionSource::Imu
                    } else if !last_failed {
                        PositionSource::Solver
                    } else if imu.is_calibrated().await {
                        // In this case our position estimate is stale, but the IMU hasn't detected enough
                        // movement so we consider the estimate as coming from the IMU.
                        PositionSource::Imu 
                    } else {
                        PositionSource::SolverStale
                    };

                    return Some(Position {
                        ra: current_ra,
                        dec: current_dec,
                        roll: current_roll,
                        source,
                        timestamp: SystemTime::now(),
                    });
                }
            }
        }

        if let Some(pos) = &mut last_solve {
            if last_failed {
                pos.source = PositionSource::SolverStale;
            }
        }

        last_solve
    }
}

// ==========================================
// MEEUS CELESTIAL TRANSFORMATIONS
// ==========================================

fn ra_dec_to_alt_az(
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    lat_deg: f64,
    lon_deg: f64,
    utc_time: chrono::DateTime<chrono::Utc>,
) -> (f64, f64, f64) {
    let y = utc_time.year() as f64;
    let m = utc_time.month() as f64;
    let d = utc_time.day() as f64;
    let h = utc_time.hour() as f64
        + utc_time.minute() as f64 / 60.0
        + utc_time.second() as f64 / 3600.0;

    let (mut y_jd, mut m_jd) = (y, m);
    if m <= 2.0 {
        y_jd -= 1.0;
        m_jd += 12.0;
    }

    let a = (y_jd / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd =
        (365.25 * (y_jd + 4716.0)).floor() + (30.6001 * (m_jd + 1.0)).floor() + d + h / 24.0 + b
            - 1524.5;

    let d_jd = jd - 2451545.0; // days since J2000.0
    let mut gmst = 280.46061837 + 360.98564736629 * d_jd;
    gmst %= 360.0;
    if gmst < 0.0 {
        gmst += 360.0;
    }

    let mut lst = gmst + lon_deg;
    lst %= 360.0;
    if lst < 0.0 {
        lst += 360.0;
    }

    let mut ha = lst - ra_deg;
    ha %= 360.0;
    if ha < 0.0 {
        ha += 360.0;
    }

    let ha_rad = ha.to_radians();
    let dec_rad = dec_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    let sin_alt = dec_rad.sin() * lat_rad.sin() + dec_rad.cos() * lat_rad.cos() * ha_rad.cos();
    let alt_rad = sin_alt.clamp(-1.0, 1.0).asin();
    let alt_deg = alt_rad.to_degrees();

    let cos_az = (dec_rad.sin() - sin_alt * lat_rad.sin()) / (alt_rad.cos() * lat_rad.cos());
    let mut az_rad = cos_az.clamp(-1.0, 1.0).acos();
    if ha_rad.sin() > 0.0 {
        az_rad = std::f64::consts::TAU - az_rad;
    }
    let az_deg = az_rad.to_degrees();

    let q_rad = (ha_rad.sin()).atan2(lat_rad.tan() * dec_rad.cos() - dec_rad.sin() * ha_rad.cos());
    let q_deg = q_rad.to_degrees();

    let alt_az_roll = roll_deg + q_deg;

    (alt_deg, az_deg, alt_az_roll)
}

fn alt_az_to_ra_dec(
    alt_deg: f64,
    az_deg: f64,
    alt_az_roll: f64,
    lat_deg: f64,
    lon_deg: f64,
    utc_time: chrono::DateTime<chrono::Utc>,
) -> (f64, f64, f64) {
    let y = utc_time.year() as f64;
    let m = utc_time.month() as f64;
    let d = utc_time.day() as f64;
    let h = utc_time.hour() as f64
        + utc_time.minute() as f64 / 60.0
        + utc_time.second() as f64 / 3600.0;

    let (mut y_jd, mut m_jd) = (y, m);
    if m <= 2.0 {
        y_jd -= 1.0;
        m_jd += 12.0;
    }

    let a = (y_jd / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd =
        (365.25 * (y_jd + 4716.0)).floor() + (30.6001 * (m_jd + 1.0)).floor() + d + h / 24.0 + b
            - 1524.5;

    let d_jd = jd - 2451545.0;
    let mut gmst = 280.46061837 + 360.98564736629 * d_jd;
    gmst %= 360.0;
    if gmst < 0.0 {
        gmst += 360.0;
    }

    let mut lst = gmst + lon_deg;
    lst %= 360.0;
    if lst < 0.0 {
        lst += 360.0;
    }

    let alt_rad = alt_deg.to_radians();
    let az_rad = az_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    let sin_dec = alt_rad.sin() * lat_rad.sin() + alt_rad.cos() * lat_rad.cos() * az_rad.cos();
    let dec_rad = sin_dec.clamp(-1.0, 1.0).asin();
    let dec_deg = dec_rad.to_degrees();

    let cos_ha = (alt_rad.sin() - sin_dec * lat_rad.sin()) / (dec_rad.cos() * lat_rad.cos());
    let mut ha_rad = cos_ha.clamp(-1.0, 1.0).acos();
    if az_rad.sin() > 0.0 {
        ha_rad = std::f64::consts::TAU - ha_rad;
    }
    let ha_deg = ha_rad.to_degrees();

    let mut ra_deg = lst - ha_deg;
    ra_deg %= 360.0;
    if ra_deg < 0.0 {
        ra_deg += 360.0;
    }

    let q_rad = (ha_rad.sin()).atan2(lat_rad.tan() * dec_rad.cos() - dec_rad.sin() * ha_rad.cos());
    let q_deg = q_rad.to_degrees();

    let eq_roll = alt_az_roll - q_deg;

    (ra_deg, dec_deg, eq_roll)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A dummy IMU for testing Custom IMU injections
    struct MockImu;
    impl ImuDevice for MockImu {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn poll_gyros(&mut self) -> Result<Vec<(nalgebra::Vector3<f64>, f64)>, String> {
            Ok(vec![])
        }
        fn revive(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_start_imu_without_location_fails() {
        // Dummy logic
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let result = fs.start_imu().await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Observer location must be set before starting the IMU."
        );
    }

    #[tokio::test]
    async fn test_start_imu_not_found() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(Some(0.0))),
            longitude: Arc::new(RwLock::new(Some(0.0))),
        };

        let result = fs.start_imu().await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Cannot start IMU because ImuType is None."
        );
    }

    #[tokio::test]
    async fn test_start_imu_double_start() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::Custom(Box::new(MockImu)))),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(Some(0.0))),
            longitude: Arc::new(RwLock::new(Some(0.0))),
        };

        let first = fs.start_imu().await;
        assert!(first.is_ok());

        let second = fs.start_imu().await;
        assert!(second.is_err());
        assert_eq!(second.unwrap_err(), "IMU is already running.");
    }

    #[tokio::test]
    async fn test_safe_stop() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let result = fs.stop_imu().await;
        assert!(result.is_ok()); // Safe to call even when not started
    }
    #[tokio::test]
    async fn test_fallback_position() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(Some(Position {
                ra: 10.0,
                dec: 20.0,
                roll: 0.0,
                source: PositionSource::Solver,
                timestamp: std::time::SystemTime::UNIX_EPOCH,
            }))),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let pos = fs.get_latest_position().await;
        assert!(pos.is_some());
        let pos = pos.unwrap();
        assert_eq!(pos.ra, 10.0);
        assert_eq!(pos.source, PositionSource::Solver);
    }

    #[tokio::test]
    async fn test_coordinate_transforms() {
        use crate::{alt_az_to_ra_dec, ra_dec_to_alt_az};
        use chrono::TimeZone;

        let time = chrono::Utc.with_ymd_and_hms(2025, 1, 1, 12, 0, 0).unwrap();
        let lat = 34.0522; // Los Angeles
        let lon = -118.2437;

        let original_ra = 45.0;
        let original_dec = 30.0;
        let original_roll = 10.0;

        let (alt, az, roll) =
            ra_dec_to_alt_az(original_ra, original_dec, original_roll, lat, lon, time);

        let (ra, dec, r_roll) = alt_az_to_ra_dec(alt, az, roll, lat, lon, time);

        assert!((original_ra - ra).abs() < 1e-6 || (original_ra - ra).abs() > 360.0 - 1e-6);
        assert!((original_dec - dec).abs() < 1e-6);
        // Roll might be shifted by 360
        assert!(
            (original_roll - r_roll).abs() < 1e-6 || (original_roll - r_roll).abs() > 360.0 - 1e-6
        );
    }

    #[tokio::test]
    async fn test_imu_fallback_when_no_anchor() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::Custom(Box::new(MockImu)))),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        fs.set_observer_location(34.0, -118.0).await;
        fs.start_imu().await.unwrap();

        let sol = tetra3::solver::Solution {
            ra: Some(100.0),
            dec: Some(50.0),
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::MatchFound,
            ..Default::default()
        };

        fs.update_anchor_from_solution(&sol, std::time::SystemTime::now())
            .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let pos = fs.get_latest_position().await;
        assert!(pos.is_some());
        let pos = pos.unwrap();
        // The IMU hasn't received any gyro data to establish a history, so update_anchor will silently fail.
        // Thus get_estimated_pointing will return Err, and we should correctly fallback to the solver's position.
        assert_eq!(pos.source, PositionSource::Solver);
    }

    #[tokio::test]
    async fn test_uninitialized_solver_fails() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)), // Solver not initialized
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        use tetra3::solver::SolveOptions;
        let options = SolveOptions::default();
        let result = fs
            .solve_from_centroids(
                &ndarray::Array2::zeros((0, 2)),
                (100.0, 100.0),
                options,
                None,
            )
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Solver is not initialized.");
    }

    #[tokio::test]
    async fn test_observer_location_updates() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        assert!(fs.latitude.read().await.is_none());
        assert!(fs.longitude.read().await.is_none());

        fs.set_observer_location(45.0, -90.0).await;

        assert_eq!(*fs.latitude.read().await, Some(45.0));
        assert_eq!(*fs.longitude.read().await, Some(-90.0));
    }

    #[tokio::test]
    async fn test_imu_status_wrappers_without_imu() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        // When IMU isn't running, these should safely return None without panicking
        assert!(fs.get_calibration_status().await.is_none());
        assert!(fs.get_motion_state().await.is_none());

        // Resetting calibration should also safely do nothing
        fs.reset_calibration().await;
    }
}
