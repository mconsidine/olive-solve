// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

#[cfg(any(target_os = "linux", target_os = "android"))]
mod hardware {
    use crate::imu::ImuDevice;
    use linux_embedded_hal::I2cdev;
    use log::{info, warn};
    use mpu6050_driver::{Address, Dlpf, FIFO_ACCEL_GYRO_FRAME_BYTES, GyroRange, Mpu6050};
    use nalgebra::Vector3;
    use std::time::{Duration, SystemTime};

    pub struct MpuXxxxDevice {
        mpu: Mpu6050<I2cdev>,
        report_interval_ms: u16,
        last_system_time: Option<SystemTime>,
    }

    impl MpuXxxxDevice {
        pub fn new(report_interval_ms: u16, addr_u8: u8) -> Result<Self, String> {
            info!(
                "Initializing MPU series hardware over I2C at address 0x{:X}...",
                addr_u8
            );
            let i2c =
                I2cdev::new("/dev/i2c-1").map_err(|e| format!("I2cdev::new failed: {:?}", e))?;

            let addr = if addr_u8 == 0x68 {
                Address::Ad0Low
            } else if addr_u8 == 0x69 {
                Address::Ad0High
            } else {
                return Err(format!("Invalid MPU address: 0x{:X}", addr_u8));
            };
            let mut mpu = Mpu6050::new(i2c, addr);

            // Wake up the device (this also validates I2C communication)
            mpu.wake()
                .map_err(|_| "Failed to wake up MPU (device not responding)".to_string())?;

            // Configure Gyro range and Digital Low Pass Filter
            mpu.set_gyro_range(GyroRange::Dps2000).ok();
            mpu.set_dlpf(Dlpf::Cfg1).ok(); // ~188Hz bandwidth

            // Configure sample rate divider for ~100Hz.
            // Cfg1 base rate is 1kHz. 1000 / (1 + 9) = 100Hz.
            mpu.set_sample_rate_divider(9).ok();

            // Reset and enable FIFO for Gyro only
            mpu.reset_fifo().ok();
            mpu.enable_motion_fifo().ok();
            mpu.enable_fifo().ok();

            Ok(Self {
                mpu,
                report_interval_ms,
                last_system_time: None,
            })
        }
    }

    impl ImuDevice for MpuXxxxDevice {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn poll_gyros(&mut self) -> Result<Vec<(Vector3<f64>, f64)>, String> {
            let mut readings = Vec::new();
            let mut frames = Vec::new();
            let mut buf = [0_u8; FIFO_ACCEL_GYRO_FRAME_BYTES];

            let fifo_count = self.mpu.fifo_count().unwrap_or(0);
            let frames_to_read = fifo_count / (FIFO_ACCEL_GYRO_FRAME_BYTES as u16);

            // Read only complete frames
            for _ in 0..frames_to_read {
                if self.mpu.read_fifo_bytes(&mut buf).is_ok() {
                    // Decode raw gyro. Note: Buf structure for MPU6050 FIFO:
                    // [Accel X, Accel Y, Accel Z, Gyro X, Gyro Y, Gyro Z]
                    // Each is 2 bytes (Big Endian)
                    let gx = i16::from_be_bytes([buf[6], buf[7]]) as f64;
                    let gy = i16::from_be_bytes([buf[8], buf[9]]) as f64;
                    let gz = i16::from_be_bytes([buf[10], buf[11]]) as f64;

                    // Convert to rad/sec based on 2000 dps scale (Scale factor: 16.4 LSB/dps)
                    let scale = 16.4;
                    let deg2rad = std::f64::consts::PI / 180.0;
                    let wx = (gx / scale) * deg2rad;
                    let wy = (gy / scale) * deg2rad;
                    let wz = (gz / scale) * deg2rad;
                    frames.push(Vector3::new(wx, wy, wz));
                }
            }

            let len = frames.len();
            if len == 0 {
                return Ok(readings);
            }

            // Back-date time logic, mirroring `bno085.rs`
            let now = SystemTime::now();
            let fallback_dt = (self.report_interval_ms as f64) / 1000.0;

            for i in 0..len {
                let steps_backward = (len - 1 - i) as u32;
                let sample_time = now
                    .checked_sub(Duration::from_secs_f64(
                        fallback_dt * (steps_backward as f64),
                    ))
                    .unwrap_or(now);

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

                readings.push((frames[i], safe_dt));
            }

            Ok(readings)
        }

        fn revive(&mut self) -> Result<(), String> {
            warn!("Sensor unresponsive. Resetting hardware FIFO...");
            self.mpu
                .reset_fifo()
                .map_err(|e| format!("Failed to reset FIFO: {:?}", e))?;
            Ok(())
        }

        fn needs_seeding(&self) -> bool {
            true
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use hardware::*;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod stub {
    use crate::imu::ImuDevice;
    use nalgebra::Vector3;

    pub struct MpuXxxxDevice;

    // Use a generic type or u8 to avoid pulling in the driver
    impl MpuXxxxDevice {
        pub fn new(_interval: u16, _address: u8) -> Result<Self, String> {
            Err("Hardware I2C is only supported on Linux/Android".into())
        }
    }

    impl ImuDevice for MpuXxxxDevice {
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
