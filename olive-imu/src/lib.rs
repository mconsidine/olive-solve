//! `olive-imu` provides abstractions and implementations for Interfacing with
//! various Inertial Measurement Units (IMUs).
//! It defines common traits and robust hardware drivers for devices such as
//! the BMI160, BNO055, and BNO085.

/// Driver for the BMI160 IMU.
pub mod bmi160;
/// Driver for the BNO055 IMU.
pub mod bno055;
/// Driver for the BNO085 IMU.
pub mod bno085;
/// Driver for the MPU Series IMUs (MPU6000, MPU6050, MPU6500, MPU9150, MPU9250).
pub mod mpuxxxx;
/// Core IMU trait, states, and data structures.
pub mod imu;
/// Storage trait for persisting IMU calibration data.
pub mod storage;

pub use bmi160::*;
pub use bno055::*;
pub use bno085::*;
pub use mpuxxxx::*;
pub use imu::*;
pub use storage::*;
