/*M!999999\- enable the sandbox mode */ 
-- MariaDB dump 10.19-11.7.2-MariaDB, for Win64 (AMD64)
--
-- Host: 192.168.0.6    Database: ImageRestoration
-- ------------------------------------------------------
-- Server version	12.0.2-MariaDB-ubu2404

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*M!100616 SET @OLD_NOTE_VERBOSITY=@@NOTE_VERBOSITY, NOTE_VERBOSITY=0 */;

--
-- Table structure for table `photo_requests`
--

DROP TABLE IF EXISTS `photo_requests`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `photo_requests` (
  `request_id` int(11) NOT NULL AUTO_INCREMENT,
  `uid` int(11) NOT NULL,
  `original_photo_path` varchar(500) NOT NULL,
  `restored_photo_path` varchar(500) DEFAULT NULL,
  `status` enum('pending','processing','done') DEFAULT 'pending',
  `request_time` timestamp NULL DEFAULT current_timestamp(),
  `restore_time` timestamp NULL DEFAULT NULL,
  `page_id` int(11) NOT NULL DEFAULT 1,
  PRIMARY KEY (`request_id`),
  KEY `fk_photo_requests_users` (`uid`),
  KEY `fk_photo_requests_service_pages` (`page_id`),
  CONSTRAINT `fk_photo_requests_service_pages` FOREIGN KEY (`page_id`) REFERENCES `service_pages` (`page_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_photo_requests_users` FOREIGN KEY (`uid`) REFERENCES `users` (`uid`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `photo_requests`
--

LOCK TABLES `photo_requests` WRITE;
/*!40000 ALTER TABLE `photo_requests` DISABLE KEYS */;
/*!40000 ALTER TABLE `photo_requests` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `service_pages`
--

DROP TABLE IF EXISTS `service_pages`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `service_pages` (
  `page_id` int(11) NOT NULL AUTO_INCREMENT,
  `page_name` varchar(100) NOT NULL,
  `page_description` text DEFAULT NULL,
  `page_url` varchar(255) NOT NULL,
  PRIMARY KEY (`page_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `service_pages`
--

LOCK TABLES `service_pages` WRITE;
/*!40000 ALTER TABLE `service_pages` DISABLE KEYS */;
/*!40000 ALTER TABLE `service_pages` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user_profiles`
--

DROP TABLE IF EXISTS `user_profiles`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_profiles` (
  `uid` int(11) NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  `phone_number` varchar(20) NOT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`uid`),
  UNIQUE KEY `phone_number` (`phone_number`),
  CONSTRAINT `fk_user_profiles_users` FOREIGN KEY (`uid`) REFERENCES `users` (`uid`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_profiles`
--

LOCK TABLES `user_profiles` WRITE;
/*!40000 ALTER TABLE `user_profiles` DISABLE KEYS */;
INSERT INTO `user_profiles` VALUES
(1,'홍길동','01012345678','2025-10-28 06:59:41');
/*!40000 ALTER TABLE `user_profiles` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `uid` int(11) NOT NULL AUTO_INCREMENT,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  PRIMARY KEY (`uid`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES
(1,'test@test.com','1234'),
(2,'test2@test.com','scrypt:32768:8:1$nGDaFIYiouZUbGgc$a127392f6c8ac9b70d8baedb8a3fb29e3417fe4e9f6cb87ab310ab2a693684824a04f6a18c3dea64e3783f61fdec58febc28fc09981ed1977d21d7737e38c766'),
(3,'test@example.com','scrypt:32768:8:1$2wGG5cO2vXmbcyMg$1f10fef0f7b098947ed97aea5f3993e33fedf42403a5ae1a96379a442cac03fe7bb16bfa1a156f070cfab336330dde21008d531c12d99bfc3dd5ce60cab406f8'),
(4,'test3@test.com','scrypt:32768:8:1$exeyKl0uVzEi2lOx$26ae121f16d3f0ddc680f27ccef2a0d57dd550ba3363d69b013ff73f649c5b80b0b89a5bc992a6bc74e270a6c397a78ccbf1e74c3e752f1a06d301cef556603c'),
(5,'1@1','scrypt:32768:8:1$6zKsK3ByQ52V9obc$5c13b94ff6c4f5792d2efa1af6f5beac461968d7010f8624434e1ea73d13375b7a0693bb0cab0b37f2af30134171cb3c0655b4d5e16828ad12df221d7bac3190');
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Temporary table structure for view `view_iser_info`
--

DROP TABLE IF EXISTS `view_iser_info`;
/*!50001 DROP VIEW IF EXISTS `view_iser_info`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8mb4;
/*!50001 CREATE VIEW `view_iser_info` AS SELECT
 1 AS `uid`,
  1 AS `email`,
  1 AS `password`,
  1 AS `name`,
  1 AS `phone_number` */;
SET character_set_client = @saved_cs_client;

--
-- Dumping routines for database 'ImageRestoration'
--

--
-- Final view structure for view `view_iser_info`
--

/*!50001 DROP VIEW IF EXISTS `view_iser_info`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_uca1400_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`%` SQL SECURITY DEFINER */
/*!50001 VIEW `imagerestoration`.`view_iser_info` AS select `u`.`uid` AS `uid`,`u`.`email` AS `email`,`u`.`password` AS `password`,`up`.`name` AS `name`,`up`.`phone_number` AS `phone_number` from (`imagerestoration`.`users` `u` join `imagerestoration`.`user_profiles` `up` on(`u`.`uid` = `up`.`uid`)) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*M!100616 SET NOTE_VERBOSITY=@OLD_NOTE_VERBOSITY */;

-- Dump completed on 2025-10-31 15:41:45
