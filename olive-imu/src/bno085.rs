// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use bno080::interface::i2c::I2cInterface;
use bno080::wrapper::BNO080;
use linux_embedded_hal::{Delay, I2cdev};
use log::{info, warn};
use nalgebra::Vector3;

use crate::imu::ImuDevice;

pub struct Bno085Device {
    imu: BNO080<I2cInterface<I2cdev>>,
    delay: Delay,
    report_interval_ms: u16,
}

impl Bno085Device {
    pub fn new(report_interval_ms: u16, address: u8) -> Result<Self, String> {
        info!(
            "Initializing BNO085 hardware over I2C at address 0x{:X}...",
            address
        );
        let i2c = I2cdev::new("/dev/i2c-1").map_err(|e| format!("I2cdev::new failed: {:?}", e))?;
        let interface = I2cInterface::new(i2c, address);
        let mut imu = BNO080::new_with_interface(interface);
        let mut delay = Delay {};

        imu.init(&mut delay)
            .map_err(|e| format!("Failed to initialize BNO085 over I2C: {:?}", e))?;

        imu.enable_gyro(report_interval_ms)
            .map_err(|e| format!("Failed to enable Calibrated Gyroscope: {:?}", e))?;

        info!(
            "Hardware initialized at {}ms using Calibrated Gyroscope.",
            report_interval_ms
        );

        Ok(Self {
            imu,
            delay,
            report_interval_ms,
        })
    }
}

impl ImuDevice for Bno085Device {
    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn poll_gyros(&mut self) -> Result<Vec<(Vector3<f64>, f64)>, String> {
        let mut readings = Vec::new();
        let hw_dt = (self.report_interval_ms as f64) / 1000.0;

        loop {
            let msg_count = self.imu.handle_one_message(&mut self.delay, 1);
            if msg_count > 0 {
                if let Ok(gyro_data) = self.imu.gyro() {
                    let wx = gyro_data[0] as f64;
                    let wy = gyro_data[1] as f64;
                    let wz = gyro_data[2] as f64;
                    readings.push((Vector3::new(wx, wy, wz), hw_dt));
                }
            } else {
                break;
            }
        }

        Ok(readings)
    }

    fn revive(&mut self) -> Result<(), String> {
        warn!("Sensor unresponsive. Sending hardware revive command...");
        self.imu
            .enable_gyro(self.report_interval_ms)
            .map_err(|e| format!("Failed to revive: {:?}", e))?;
        Ok(())
    }
}
