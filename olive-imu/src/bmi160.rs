// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use bmi160::{
    Bmi160, GyroscopePowerMode, GyroscopeRange, SensorSelector, SlaveAddr, interface::I2cInterface,
};
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

        info!("BMI160 initialized. Gyroscope running at 2000 deg/s range.");
        Ok(())
    }

    fn poll_gyros(&mut self) -> Result<Vec<(Vector3<f64>, f64)>, String> {
        // We select the gyro and the hardware sensor time
        let selector = SensorSelector::new().gyro().time();
        match self.imu.data_scaled(selector) {
            Ok(data) => {
                if let (Some(gyro), Some(current_time)) = (data.gyro, data.time) {
                    // data_scaled returns values in deg/sec
                    // We need to convert them to radians/sec for the Imu abstraction
                    let deg_to_rad = std::f64::consts::PI / 180.0;

                    let wx = gyro.x as f64 * deg_to_rad;
                    let wy = gyro.y as f64 * deg_to_rad;
                    let wz = gyro.z as f64 * deg_to_rad;

                    let dt = if let Some(last_time) = self.last_sensor_time {
                        // SENSORTIME is a 24-bit counter with 39us resolution
                        let mut diff = current_time as i64 - last_time as i64;
                        if diff < 0 {
                            diff += 0x1000000;
                        }
                        (diff as f64) * 39.0e-6
                    } else {
                        0.01 // Initial fallback dt
                    };

                    self.last_sensor_time = Some(current_time);

                    Ok(vec![(Vector3::new(wx, wy, wz), dt)])
                } else {
                    Ok(Vec::new())
                }
            }
            Err(_) => {
                // Return empty vec on transient read errors rather than crashing the system.
                // The Imu watchdog will handle revive() if it drops too many packets.
                Ok(Vec::new())
            }
        }
    }

    fn revive(&mut self) -> Result<(), String> {
        warn!("BMI160 unresponsive. Sending hardware revive command...");
        // Re-assert power mode in an attempt to wake up the sensor
        self.imu
            .set_gyro_power_mode(GyroscopePowerMode::Normal)
            .map_err(|_| "Failed to revive BMI160 gyro".to_string())?;

        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(())
    }
}
