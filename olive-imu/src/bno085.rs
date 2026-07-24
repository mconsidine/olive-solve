// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

#[cfg(any(target_os = "linux", target_os = "android"))]
mod hardware {
    use bno080::interface::i2c::I2cInterface;
    use bno080::wrapper::BNO080;
    use linux_embedded_hal::{Delay, I2cdev};
    use log::{info, warn};
    use nalgebra::Vector3;

    use std::time::{Duration, SystemTime};

    use crate::imu::ImuDevice;

    pub struct Bno085Device {
        imu: BNO080<I2cInterface<I2cdev>>,
        delay: Delay,
        report_interval_ms: u16,
        use_calibrated: bool,
        last_system_time: Option<SystemTime>,
    }

    impl Bno085Device {
        pub fn new(
            report_interval_ms: u16,
            address: u8,
            use_calibrated: bool,
        ) -> Result<Self, String> {
            info!(
                "Initializing BNO085 hardware over I2C at address 0x{:X}...",
                address
            );
            let i2c =
                I2cdev::new("/dev/i2c-1").map_err(|e| format!("I2cdev::new failed: {:?}", e))?;
            let interface = I2cInterface::new(i2c, address);
            let mut imu = BNO080::new_with_interface(interface);
            let mut delay = Delay {};

            imu.init(&mut delay)
                .map_err(|e| format!("Failed to initialize BNO085 over I2C: {:?}", e))?;

            let mode_str = if use_calibrated {
                "Calibrated"
            } else {
                "Uncalibrated"
            };

            if use_calibrated {
                imu.enable_gyro_calibrated(report_interval_ms)
                    .map_err(|e| format!("Failed to enable Calibrated Gyroscope: {:?}", e))?;
            } else {
                imu.enable_gyro(report_interval_ms)
                    .map_err(|e| format!("Failed to enable Uncalibrated Gyroscope: {:?}", e))?;
            }

            info!(
                "Hardware initialized at {}ms using {} Gyroscope.",
                report_interval_ms, mode_str
            );

            Ok(Self {
                imu,
                delay,
                report_interval_ms,
                use_calibrated,
                last_system_time: None,
            })
        }
    }

    impl ImuDevice for Bno085Device {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn poll_gyros(&mut self) -> Result<Vec<(Vector3<f64>, f64)>, String> {
            let _msg_count = self.imu.handle_all_messages(&mut self.delay, 1);

            let mut readings = Vec::new();

            let (len, queue) = if self.use_calibrated {
                self.imu.calibrated_gyro_queue()
            } else {
                self.imu.gyro_queue()
            };

            if len == 0 {
                return Ok(readings);
            }

            // Since we are polling over I2C without a hardware interrupt (HINT) pin, the BNO085's
            // internal timestamps reset on packet boundaries, making them unusable for absolute time.
            // Instead, we use "back-dating": we anchor the *last* sample in the queue to the host's
            // current wall-clock time (`now`), and step backwards by the requested hardware interval
            // for each preceding sample. This forces the boundary sample to absorb any I2C loop jitter
            // keeping the integration timeline aligned with real-world physical time.
            let now = SystemTime::now();
            let fallback_dt = (self.report_interval_ms as f64) / 1000.0;

            for i in 0..len {
                let steps_backward = (len - 1 - i) as u32;
                let sample_time = now
                    .checked_sub(Duration::from_secs_f64(
                        fallback_dt * (steps_backward as f64),
                    ))
                    .unwrap_or(now);

                let (_timestamp, gyro_data) = queue[i];
                let wx = gyro_data[0] as f64;
                let wy = gyro_data[1] as f64;
                let wz = gyro_data[2] as f64;

                let dt = if let Some(last) = self.last_system_time {
                    sample_time
                        .duration_since(last)
                        .unwrap_or(Duration::from_secs_f64(fallback_dt))
                        .as_secs_f64()
                } else {
                    fallback_dt
                };

                let safe_dt = if dt <= 0.0 { fallback_dt } else { dt };

                self.last_system_time = Some(sample_time);

                readings.push((Vector3::new(wx, wy, wz), safe_dt));
            }

            Ok(readings)
        }

        fn revive(&mut self) -> Result<(), String> {
            warn!("Sensor unresponsive. Sending hardware revive command...");
            if self.use_calibrated {
                self.imu
                    .enable_gyro_calibrated(self.report_interval_ms)
                    .map_err(|e| format!("Failed to revive: {:?}", e))?;
            } else {
                self.imu
                    .enable_gyro(self.report_interval_ms)
                    .map_err(|e| format!("Failed to revive: {:?}", e))?;
            }
            Ok(())
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use hardware::*;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod stub {
    use crate::imu::ImuDevice;
    use nalgebra::Vector3;

    pub struct Bno085Device;

    impl Bno085Device {
        pub fn new(_interval: u16, _address: u8, _calib: bool) -> Result<Self, String> {
            Err("Hardware I2C is only supported on Linux/Android".into())
        }
    }

    impl ImuDevice for Bno085Device {
        fn init(&mut self) -> Result<(), String> {
            Err("Unsupported".into())
        }
        fn poll_gyros(&mut self) -> Result<Vec<(Vector3<f64>, f64)>, String> {
            Err("Unsupported".into())
        }
        fn revive(&mut self) -> Result<(), String> {
            Err("Unsupported".into())
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub use stub::*;
