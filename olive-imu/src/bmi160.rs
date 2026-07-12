// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use bmi160::{Bmi160, GyroscopePowerMode, GyroscopeRange, SlaveAddr, interface::I2cInterface};
use linux_embedded_hal::I2cdev;
use log::{info, warn};
use nalgebra::Vector3;

use crate::imu::ImuDevice;

pub struct Bmi160Device {
    imu: Bmi160<I2cInterface<I2cdev>>,
    last_sensor_time: Option<u32>,
}

impl Bmi160Device {
    pub fn new(address_u8: u8) -> Result<Self, String> {
        info!(
            "Initializing BMI160 hardware over I2C at address 0x{:X}...",
            address_u8
        );
        let i2c = I2cdev::new("/dev/i2c-1").map_err(|e| format!("I2cdev::new failed: {:?}", e))?;
        let address = if address_u8 == 0x69 {
            SlaveAddr::Alternative(true)
        } else {
            SlaveAddr::Default
        };
        let imu = Bmi160::new_with_i2c(i2c, address);

        Ok(Self {
            imu,
            last_sensor_time: None,
        })
    }
}

impl ImuDevice for Bmi160Device {
    fn init(&mut self) -> Result<(), String> {
        // Set gyro range to 2000 deg/s for high movement applications
        self.imu
            .set_gyro_range(GyroscopeRange::Scale2000)
            .map_err(|_| "Failed to set BMI160 gyro range".to_string())?;

        // Turn on the gyro
        self.imu
            .set_gyro_power_mode(GyroscopePowerMode::Normal)
            .map_err(|_| "Failed to enable BMI160 gyro".to_string())?;

        // BMI160 needs ~100ms for gyro to fully turn on from suspend
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Configure FIFO for Header mode + Gyro + Time
        self.imu
            .config_fifo()
            .map_err(|_| "Failed to configure BMI160 FIFO".to_string())?;

        info!(
            "BMI160 initialized. Gyroscope running at 2000 deg/s range with hardware FIFO enabled."
        );
        Ok(())
    }

    fn poll_gyros(&mut self) -> Result<Vec<(Vector3<f64>, f64)>, String> {
        let mut buffer = [0u8; 1024];
        let mut readings = Vec::new();

        match self.imu.read_fifo(&mut buffer) {
            Ok(len) if len > 0 => {
                // DEBUG LOGGING
                // log::info!("FIFO read length: {}. First bytes: {:?}", len, &buffer[1..core::cmp::min(len+1, 10)]);

                let mut i = 1; // FIFO data starts at index 1 because index 0 was the register address
                let mut gyro_frames = Vec::new();
                let mut current_sensor_time: Option<u32> = None;

                while i <= len {
                    let header = buffer[i];
                    i += 1;

                    // We might need to match 0x84 or 0x88 for Gyro, let's match both temporarily to test
                    if header == 0x84 || header == 0x88 {
                        // Gyro frame (Header + 6 bytes)
                        if i + 6 <= len + 1 {
                            // Data in FIFO is little-endian: LSB, MSB
                            let gx = (buffer[i] as u16 | ((buffer[i + 1] as u16) << 8)) as i16;
                            let gy = (buffer[i + 2] as u16 | ((buffer[i + 3] as u16) << 8)) as i16;
                            let gz = (buffer[i + 4] as u16 | ((buffer[i + 5] as u16) << 8)) as i16;

                            let deg_to_rad = std::f64::consts::PI / 180.0;
                            // 2000 deg/s range = 16.4 LSB per deg/s
                            let wx = (gx as f64 / 16.4) * deg_to_rad;
                            let wy = (gy as f64 / 16.4) * deg_to_rad;
                            let wz = (gz as f64 / 16.4) * deg_to_rad;

                            gyro_frames.push(Vector3::new(wx, wy, wz));
                            i += 6;
                        } else {
                            break; // Malformed / cut off
                        }
                    } else if header == 0x44 {
                        // Sensortime frame (Header + 3 bytes)
                        if i + 3 <= len + 1 {
                            let time = (buffer[i] as u32)
                                | ((buffer[i + 1] as u32) << 8)
                                | ((buffer[i + 2] as u32) << 16);
                            current_sensor_time = Some(time);
                            i += 3;
                        } else {
                            break;
                        }
                    } else if header == 0x40 {
                        // Skip frame (Header + 1 byte)
                        i += 1;
                    } else if header == 0x80 {
                        // Empty / Invalid (end of valid data)
                        break;
                    } else {
                        // Unknown header (maybe Accel/Mag was enabled by mistake or misalignment)
                        // Safety break to prevent infinite loops or garbage data
                        log::warn!("BMI160 unknown header: 0x{:X} at index {}", header, i - 1);
                        break;
                    }
                }

                // SENSORTIME is an absolute, continuous 24-bit hardware counter with 39us resolution.
                // Because the BMI160 generates this internally and independently, it is immune to I2C jitter.
                // We simply calculate the exact hardware ticks elapsed since the last batch, handle 24-bit
                // wraparounds, and evenly distribute that total physical time across the samples in the batch.
                let total_dt = if let Some(time) = current_sensor_time {
                    let dt = if let Some(last_time) = self.last_sensor_time {
                        // SENSORTIME is a 24-bit counter with 39us resolution
                        let mut diff = time as i64 - last_time as i64;
                        if diff < -8_000_000 {
                            // Massive negative drop means it wrapped around 0x1000000
                            diff += 0x1000000;
                        } else if diff < 0 {
                            // Tiny negative drop means jitter / duplicate
                            diff = 0;
                        }
                        (diff as f64) * 39.0e-6
                    } else {
                        // Initial fallback: assume 10ms per frame (for 100Hz default ODR)
                        0.01 * (gyro_frames.len() as f64).max(1.0)
                    };
                    self.last_sensor_time = Some(time);
                    dt
                } else {
                    // Fallback if sensortime was missing
                    0.01 * (gyro_frames.len() as f64).max(1.0)
                };

                if !gyro_frames.is_empty() {
                    let avg_dt = total_dt / (gyro_frames.len() as f64);
                    for frame in gyro_frames {
                        readings.push((frame, avg_dt));
                    }
                }

                Ok(readings)
            }
            Ok(_) => Ok(Vec::new()), // len == 0
            Err(e) => {
                log::warn!("BMI160 FIFO read error: {:?}", e);
                // Return empty vec on transient read errors rather than crashing the system.
                // The Imu watchdog will handle revive() if it drops too many packets.
                Ok(Vec::new())
            }
        }
    }

    fn revive(&mut self) -> Result<(), String> {
        warn!("BMI160 unresponsive. Sending hardware revive command...");
        // Re-assert power mode and reconfigure FIFO in an attempt to wake up the sensor
        self.imu
            .set_gyro_power_mode(GyroscopePowerMode::Normal)
            .map_err(|_| "Failed to revive BMI160 gyro".to_string())?;

        let _ = self.imu.config_fifo();

        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(())
    }
}
