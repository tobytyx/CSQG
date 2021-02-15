CREATE TABLE IF NOT EXISTS `user`(
   `email` VARCHAR(255) primary key,
   `username` VARCHAR(255),
   `password` VARCHAR(255)
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `guesswhat_record`(
   `record_id` VARCHAR(255) primary key,
   `username` VARCHAR(255),
   `create_time` DATETIME default current_timestamp,
   `img_name` VARCHAR(255),
   `history` TEXT,
   `guess` INT
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
