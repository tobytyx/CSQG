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


INSERT INTO `guesswhat_record` (`record_id`, `username`, `create_time`, `img_name`, `history`, `guess`)
VALUES
	('58a587bd-cdc8-44c5-8c7c-1ba27e4ac940','sssn','2021-02-10 10:35:03','COCO_train2014_000000386864.jpg','is it a dog ?	Yes\nis it the dog on the left ?	No\nis it the dog on the right ?	Yes\nis it the whole dog ?	Yes\nis it the whole dog ?	Yes',0),
	('8b363302-444e-41ff-a677-e54786faed7e','sssn','2021-02-14 17:25:52','COCO_train2014_000000037160.jpg','is it a horse ?	No\nis it a person ?	Yes\non the person on the horse ?	No\non the left ?	Yes\nthe whole white ?	Yes',1),
	('cdca9596-adb6-499f-9fae-6948f2d527c7','sssn','2021-02-14 17:24:14','COCO_train2014_000000231572.jpg','is it a person ?	Yes\nare they on the foreground ?	No\nare they on the left ?	No\nare they on the right ?	Yes\nare they on the middle ?	No',1);
