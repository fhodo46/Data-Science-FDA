#Section 1- Installation and Loading of Libraries
cat("Step 1: Installing packages...\n")

install.packages("ggplot2")
install.packages("dplyr")
install.packages("randomForest")
install.packages("caret")
install.packages("tidyr")

library(ggplot2)
library(dplyr)
library(randomForest)
library(caret)
library(tidyverse)

cat("Step 1 done!\n")

#Section 2- Loading the data
cat("Step 2: Loading data...\n")

df <- read.csv("C:/Users/DELL/Downloads/food_delivery_data.csv", stringsAsFactors = FALSE)
cat("Rows:", nrow(df), "| Columns:", ncol(df), "\n")
cat("Step 2 done!\n")

#Section 3- Data Preprocessing
cat("Step 3: Preprocessing data...\n")

df$Weatherconditions <- trimws(gsub("conditions ", "", df$Weatherconditions))
df$Time_taken <- as.numeric(gsub("\\(min\\) ", "", df$Time_taken.min.))
df <- df %>% mutate(across(where(is.character), trimws))
df$multiple_deliveries <- as.numeric(df$multiple_deliveries)
df <- df %>% filter(!is.na(Weatherconditions), Weatherconditions!="NaN", Weatherconditions!="", 
!is.na(Time_taken), Time_taken!="NaN", Time_taken!="", 
!is.na(Delivery_person_Age), Delivery_person_Age!="NaN", Delivery_person_Age!="", 
!is.na(Delivery_person_Ratings), Delivery_person_Ratings!="NaN", Delivery_person_Age!="",
!is.na(multiple_deliveries), multiple_deliveries!="NaN", multiple_deliveries!="")

#Calculation of Haversine distance
haversine <- function(lat1, lon1, lat2, lon2) {
  R <- 6371
  dlat <- (lat2 - lat1) * pi / 180
  dlon <- (lon2 - lon1) * pi / 180
  a <- sin(dlat/2)^2 + cos(lat1*pi/180)*cos(lat2*pi/180)*sin(dlon/2)^2
  R * 2 * atan2(sqrt(a), sqrt(1-a))
}

df$distance_km <- haversine(df$Restaurant_latitude, df$Restaurant_longitude,
                             df$Delivery_location_latitude, df$Delivery_location_longitude)
df <- df %>% filter(distance_km > 0.1, distance_km < 100)

cat("Clean rows:", nrow(df), "\n")
cat("Step 3 done!\n")

#Section 4 - Exploratory Data Analysis
cat("Step 4: Creating plots...\n")

#Plot 1 - Distribution of Delivery Time
p1 <- ggplot(df, aes(x = Time_taken)) +
  geom_histogram(binwidth = 2, fill = "#2196F3", color = "white") +
  labs(title = "Distribution of Delivery Time", x = "Time (min)", y = "Count") 


#Plot 2 - Weather vs Time
p2 <- ggplot(df, aes(x = reorder(Weatherconditions, Time_taken, median), y = Time_taken, fill = Weatherconditions)) +
  geom_boxplot(alpha = 0.8) + coord_flip() +
  labs(title = "Delivery Time by Weather", x = "Weather", y = "Time (min)")

#Plot 3 - Traffic vs Time
df$Road_traffic_density <- factor(df$Road_traffic_density, levels = c("Low","Medium","High","Jam"))
p3 <- ggplot(df, aes(x = Road_traffic_density, y = Time_taken, fill = Road_traffic_density)) +
  geom_boxplot(alpha = 0.8) +
  labs(title = "Delivery Time by Traffic", x = "Traffic", y = "Time (min)")

#Plot 4 - Distance vs Time
p4 <- ggplot(df %>% sample_n(100), aes(x = distance_km, y = Time_taken)) +
  geom_point(alpha = 0.9, color = "#E91E63") +
  geom_smooth(method = "lm", color = "#1A237E", se = FALSE) +
  labs(title = "Distance vs Delivery Time", x = "Distance (km)", y = "Time (min)")

#Descriptive stats
cat("\n--- Descriptive Statistics ---\n")
cat("Mean delivery time:", round(mean(df$Time_taken), 2), "min\n")
cat("Median:", median(df$Time_taken), "min\n")
cat("Std Dev:", round(sd(df$Time_taken), 2), "\n")
cat("Correlation Distance-Time:", round(cor(df$distance_km, df$Time_taken), 3), "\n")
cat("Step 4 done!\n")

#Section 5- Random Forest Model
cat("Step 5: Building model (2-3 minutes)...\n")

model_df <- df %>%
  select(Time_taken, Delivery_person_Age, Delivery_person_Ratings,
         distance_km, multiple_deliveries, Vehicle_condition,
         Weatherconditions, Road_traffic_density, Type_of_vehicle, City) %>%
  mutate(across(where(is.character),
                function(x) {
                  x <- na_if(x, "")
                  x <- na_if(x, "NaN")
                  x
                })) %>%
  drop_na() %>%
  mutate(across(where(is.character), as.factor))
  
# Use only 10,000 rows for speed
set.seed(42)
model_df <- model_df %>% sample_n(min(10000, nrow(model_df)))

trainIdx  <- createDataPartition(model_df$Time_taken, p = 0.8, list = FALSE)
train_data <- model_df[trainIdx, ]
test_data  <- model_df[-trainIdx, ]

cat("Training rows:", nrow(train_data), "| Test rows:", nrow(test_data), "\n")

rf_model <- randomForest(Time_taken ~ ., data = train_data, ntree = 50, mtry = 3, importance = TRUE)
cat("Model trained!\n")

predictions <- predict(rf_model, test_data)

rmse_val <- sqrt(mean((test_data$Time_taken - predictions)^2))
mae_val  <- mean(abs(test_data$Time_taken - predictions))
r2_val   <- 1 - sum((test_data$Time_taken - predictions)^2) /
                sum((test_data$Time_taken - mean(test_data$Time_taken))^2)

cat("\n--- Model Evaluation ---\n")
cat("RMSE:", round(rmse_val, 3), "\n")
cat("MAE: ", round(mae_val,  3), "\n")
cat("R2:  ", round(r2_val,   3), "\n")

#Plot 5- Feature Importance
imp <- as.data.frame(importance(rf_model))
imp$Feature <- rownames(imp)
p5 <- ggplot(imp, aes(x = reorder(Feature, `%IncMSE`), y = `%IncMSE`)) +
  geom_col(fill = "#3F51B5") + coord_flip() +
  labs(title = "Feature Importance", x = "Feature", y = "% Increase MSE")

#Section 6- Summary
cat("\n=== Analysis Complete ===\n")
cat("Average delivery time:", round(mean(df$Time_taken), 1), "min\n")
cat("R2 =", round(r2_val, 3), "- model explains", paste0(round(r2_val*100, 1), "%"), "of variance\n")
idx <- sample(1:nrow(test_data), 100)
actual <- test_data$Time_taken[idx]
predicted <- predictions[idx]

plot(predicted, actual, col = "blue", pch = 16, xlab = "Predicted", ylab = "Actual", main = "Predicted vs Actual (100 samples)")
abline(0, 1, col = "red", lwd = 2)