library(tidyverse)
library(R.utils)
library(extRemes)
library(lmom)
library(Kendall)
library(ggplot2)
library(maps)
library(revdbayes)
library(scales)

# Part 1: File Setup
# Download files
base_url <- "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

# Destination folder
files_path <- # Insert file path

# Get directory listing and extract file names
page <- readLines(base_url)

files <- grep("StormEvents_details.*\\.csv\\.gz", page, value = TRUE) |>
  str_extract("StormEvents_details-ftp_v1\\.0_d\\d{4}_c\\d+\\.csv\\.gz") |>
  na.omit()

walk(files, function(f) {
  dest <- paste0(files_path, "/", f)
  download.file(paste0(base_url, f), destfile = dest, mode = "wb")
  message("Downloaded: ", f)
})

# Find all .gz files in folder
gz_files <- list.files(files_path,
                       pattern = "\\.gz$",
                       full.names = TRUE)

# Unzip each one to the same folder
walk(gz_files, function(f) {
  out_file <- file.path(files_path, gsub("\\.gz$", "", basename(f)))
  if (!file.exists(out_file)) {
    R.utils::gunzip(f, destname = out_file, remove = FALSE)
    message("Unzipped: ", basename(f))
  } else {
    message("Already unzipped: ", basename(f))
  }
})

# Part 2: Data Setup

# Filter for hail, desired years, and extract desired columns
hail_data_1Plus <- list.files(files_path,
                        pattern = "StormEvents_details", 
                        full.names = TRUE) |>
  map_dfr(~read_csv(.x, show_col_types = FALSE, 
                    col_types = cols(.default = "c"))) |>
  filter(EVENT_TYPE == "Hail",
         MAGNITUDE >= 1.0, #Severe hail criterion
         YEAR >= "1950", YEAR <= "2025",
         CZ_FIPS != 0,
         CZ_TYPE == "C") |>
  select(YEAR, MONTH_NAME, STATE, CZ_NAME, CZ_FIPS,
         CZ_TYPE, MAGNITUDE, DAMAGE_PROPERTY, DAMAGE_CROPS) |>
  mutate(
    YEAR      = as.integer(YEAR),
    CZ_FIPS   = as.integer(CZ_FIPS),
    MAGNITUDE = as.numeric(MAGNITUDE)
  )

# Convert NOAA damage strings to numeric
parse_damage <- function(x) {
  x <- str_trim(x)
  multiplier <- case_when(
    str_detect(x, "K") ~ 1e3,
    str_detect(x, "M") ~ 1e6,
    str_detect(x, "B") ~ 1e9,
    TRUE ~ 1
  )
  as.numeric(str_remove_all(x, "[KMBkmb]")) * multiplier
}

# Apply numeric changes to damages to property and crops
hail_data_1Plus <- hail_data_1Plus |>
  mutate(
    DAMAGE_PROPERTY = parse_damage(DAMAGE_PROPERTY),
    DAMAGE_CROPS    = parse_damage(DAMAGE_CROPS),
    DAMAGE_PROPERTY = replace_na(DAMAGE_PROPERTY, 0),
    DAMAGE_CROPS    = replace_na(DAMAGE_CROPS, 0)
  )


# Part 3: Group data & Basic Summary

# Hail diameter 1-2 inches
hail_data_1_2 <- hail_data_1Plus |>
  filter(MAGNITUDE < 2.0)

# Hail diameter 2-3 inches
hail_data_2_3 <- hail_data_1Plus |>
  filter(MAGNITUDE >= 2.0 & MAGNITUDE < 3.0)

# Hail diameter 3-4 inches
hail_data_3_4 <- hail_data_1Plus |>
  filter(MAGNITUDE >= 3.0 & MAGNITUDE < 4.0)

# Hail diameter 4+ inches
hail_data_4Plus <- hail_data_1Plus |>
  filter(MAGNITUDE >= 4.0)

# Basic Statistical Summary Table
summary_df <- data.frame(
  Hail_Size   = c("1.0+ in", 
                  "1.0-2.0 in", 
                  "2.0-3.0 in", 
                  "3.0-4.0 in", 
                  "4.0+ in"),
  N_Reports   = c(
    nrow(hail_data_1Plus),
    nrow(hail_data_1_2),
    nrow(hail_data_2_3),
    nrow(hail_data_3_4),
    nrow(hail_data_4Plus)
  ),
  Prop_Mean   = c(
    mean(hail_data_1Plus$DAMAGE_PROPERTY, na.rm = TRUE),
    mean(hail_data_1_2$DAMAGE_PROPERTY, na.rm = TRUE),
    mean(hail_data_2_3$DAMAGE_PROPERTY, na.rm = TRUE),
    mean(hail_data_3_4$DAMAGE_PROPERTY, na.rm = TRUE),
    mean(hail_data_4Plus$DAMAGE_PROPERTY, na.rm = TRUE)
  ),
  Prop_Max    = c(
    max(hail_data_1Plus$DAMAGE_PROPERTY, na.rm = TRUE),
    max(hail_data_1_2$DAMAGE_PROPERTY, na.rm = TRUE),
    max(hail_data_2_3$DAMAGE_PROPERTY, na.rm = TRUE),
    max(hail_data_3_4$DAMAGE_PROPERTY, na.rm = TRUE),
    max(hail_data_4Plus$DAMAGE_PROPERTY, na.rm = TRUE)
  ),
  Crop_Mean   = c(
    mean(hail_data_1Plus$DAMAGE_CROPS, na.rm = TRUE),
    mean(hail_data_1_2$DAMAGE_CROPS, na.rm = TRUE),
    mean(hail_data_2_3$DAMAGE_CROPS, na.rm = TRUE),
    mean(hail_data_3_4$DAMAGE_CROPS, na.rm = TRUE),
    mean(hail_data_4Plus$DAMAGE_CROPS, na.rm = TRUE)
  ),
  Crop_Max    = c(
    max(hail_data_1Plus$DAMAGE_CROPS, na.rm = TRUE),
    max(hail_data_1_2$DAMAGE_CROPS, na.rm = TRUE),
    max(hail_data_2_3$DAMAGE_CROPS, na.rm = TRUE),
    max(hail_data_3_4$DAMAGE_CROPS, na.rm = TRUE),
    max(hail_data_4Plus$DAMAGE_CROPS, na.rm = TRUE)
  )
)

# Format for readability
fmt_dollar <- function(x) {
  case_when(
    x >= 1e9  ~ paste0("$", round(x / 1e9, 1), "B"),
    x >= 1e6  ~ paste0("$", round(x / 1e6, 1), "M"),
    x >= 1e3  ~ paste0("$", round(x / 1e3, 1), "K"),
    TRUE      ~ paste0("$", round(x, 0))
  )
}

summary_df_formatted <- summary_df |>
  mutate(
    Prop_Mean   = fmt_dollar(Prop_Mean),
    Prop_Max    = fmt_dollar(Prop_Max),
    Crop_Mean   = fmt_dollar(Crop_Mean),
    Crop_Max    = fmt_dollar(Crop_Max),
    N_Reports   = scales::comma(N_Reports)
  )

print(summary_df_formatted)

# Compute Annual Maximum Hail Size Per County
annual_max <- hail_data_1Plus |>
  group_by(CZ_FIPS, CZ_NAME, STATE, YEAR) |>
  summarise(max_hail = max(MAGNITUDE, na.rm = TRUE), .groups = "drop") |>
  mutate(county_id = paste(CZ_NAME, STATE, sep = "_"))

# Counties with at least 10 years of data
sufficient_counties <- annual_max |>
  group_by(county_id) |>
  summarise(years_with_hail = n_distinct(YEAR)) |>
  filter(years_with_hail >= 10)

# County id is necessary to distinguish same name counties in different states
annual_max <- annual_max |>
  semi_join(sufficient_counties, by = "county_id")

# Remove duplicates and non-existent mapping polygons
annual_max <- annual_max |>
  filter(!(CZ_NAME == "DISTRICT OF COLUMBIA" & STATE == "DISTRICT OF COLUMBIA")) |>
  filter(!(CZ_NAME == "CHESAPEAKE (C)"       & STATE == "VIRGINIA")) |>
  filter(!(CZ_NAME == "ST. LOUIS (C)"        & STATE == "MISSOURI")) |>
  filter(!(CZ_NAME == "RICHMOND (C)"         & STATE == "VIRGINIA"))

# list of all counties by year with maximum annual hail size 
county_list <- split(annual_max, annual_max$county_id)

# County coverage map: county_list
# Get U.S. county map data
us_counties <- map_data("county")

# Create a lookup of counties in our analysis
# map_data uses lowercase state and county names
county_coverage <- annual_max |>
  mutate(
    state_lower  = tolower(STATE),
    county_lower = tolower(CZ_NAME)
  ) |>
  distinct(state_lower, county_lower) |>
  mutate(in_analysis = TRUE)

# clean county names
clean_county_name <- function(x) {
  x <- tolower(x)
  x <- str_remove(x, " \\(c\\)")
  x <- str_replace_all(x, "st\\.", "st")
  x <- str_replace_all(x, "ste\\.", "ste")
  x <- str_replace_all(x, "o'brien", "obrien")
  x <- str_replace_all(x, "o brien", "o'brien")
  x <- str_replace_all(x, "dekalb", "de kalb")
  x <- str_replace_all(x, "prince george's", "prince georges")
  x <- str_replace_all(x, "queen anne's", "queen annes")
  x <- str_replace_all(x, "st mary's", "st marys")
  x <- str_trim(x)
  return(x)
}

# Apply to county_coverage
county_coverage <- annual_max |>
  mutate(
    state_lower  = tolower(STATE),
    county_lower = clean_county_name(CZ_NAME)
  ) |>
  distinct(state_lower, county_lower) |>
  mutate(in_analysis = TRUE)

# Join with map data
us_counties <- us_counties |>
  left_join(county_coverage, 
            by = c("region" = "state_lower", 
                   "subregion" = "county_lower")) |>
  mutate(in_analysis = ifelse(is.na(in_analysis), FALSE, TRUE))

# Get state outlines for overlay
us_states <- map_data("state")

# Number of counties shaded
n_shaded <- us_counties |>
  filter(in_analysis == TRUE) |>
  distinct(region, subregion) |>
  nrow()

# Plot
ggplot() +
  geom_polygon(data = us_counties,
               aes(x = long, y = lat, group = group, 
                   fill = in_analysis),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_manual(values = c("FALSE" = "grey90", "TRUE" = "steelblue"),
                    labels = c("Not Included", "Included in Analysis"),
                    name = "") +
  coord_fixed(1.3) +
  labs(
    title = "U.S. Counties Included in Hail Risk Analysis",
    subtitle = paste0("Counties with ≥10 years of significant hail records (1950-2024), n = ",
                      n_shaded),
    caption = "Source: NOAA Storm Events Database"
  ) +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "bottom"
  )

# Part 4: Mann-Kendall Test to assess stationarity vs nonstationarity

mk_results <- map_dfr(county_list, function(df) {
  
  # Sort by year to ensure correct time ordering
  df    <- df |> arrange(YEAR)
  data  <- df$max_hail
  name  <- unique(df$CZ_NAME)
  state <- unique(df$STATE)
  fips  <- unique(df$CZ_FIPS)[1]
  
  # Perform test
  mk <- MannKendall(data)
  
  # Results
 mk_data_summary <- data.frame(
    CZ_FIPS   = fips,
    CZ_NAME   = name,
    STATE     = state,
    mk_tau    = as.numeric(mk$tau),
    mk_pval   = as.numeric(mk$sl),
    trend     = ifelse(mk$sl < 0.05, "Significant", "Not Significant"),
    direction = ifelse(mk$tau > 0, "Increasing", "Decreasing")
  )
})

# Remove duplicates
mk_results <- mk_results |>
  distinct(CZ_NAME, STATE, .keep_all = TRUE)

# County summary
cat("Counties tested:", nrow(mk_results), "\n")
cat("Significant trends:", sum(mk_results$trend == "Significant"), "\n")
cat("Not significant:", sum(mk_results$trend == "Not Significant"), "\n")

# Summary of trend directions among significant counties
mk_results |>
  filter(trend == "Significant") |>
  count(direction)

# Updated U.S. counties map

# Get U.S. county map data
us_counties <- map_data("county")

# Prepare Mann-Kendall results - combine trend and direction
mk_map <- mk_results |>
  mutate(
    state_lower  = tolower(STATE),
    county_lower = clean_county_name(tolower(CZ_NAME)),
    trend_status = case_when(
      trend == "Significant" & direction == "Increasing" ~ "Nonstationary Increasing",
      trend == "Significant" & direction == "Decreasing" ~ "Nonstationary Decreasing",
      trend == "Not Significant"                         ~ "Stationary",
    )
  ) |>
  distinct(state_lower, county_lower, .keep_all = TRUE)  # remove duplicates

# Join with map data
us_counties <- us_counties |>
  left_join(mk_map,
            by = c("region"    = "state_lower",
                   "subregion" = "county_lower")) |>
  mutate(trend_status = ifelse(is.na(trend_status), "No Data", trend_status))

# Get state outlines
us_states <- map_data("state")

# Count shaded counties by trend status
shaded_trend <- us_counties |>
  filter(trend_status != "No Data") |>
  distinct(region, subregion, .keep_all = TRUE) |>
  count(trend_status)

# Extract counts for subtitle
n_stationary = shaded_trend$n[shaded_trend$trend_status == "Stationary"]
n_increasing = shaded_trend$n[shaded_trend$trend_status == "Nonstationary Increasing"]
n_decreasing = shaded_trend$n[shaded_trend$trend_status == "Nonstationary Decreasing"]

# Plot
ggplot() +
  geom_polygon(data = us_counties,
               aes(x = long, y = lat, group = group,
                   fill = trend_status),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_manual(
    values = c("Nonstationary Increasing" = "firebrick",
               "Nonstationary Decreasing" = "steelblue",
               "Stationary"               = "grey",
               "No Data"                  = "grey90"),
    labels = c("Nonstationary Increasing" = "Nonstationary - Increasing (p < 0.05)",
               "Nonstationary Decreasing" = "Nonstationary - Decreasing (p < 0.05)",
               "Stationary"               = "Stationary (p ≥ 0.05)",
               "No Data"                  = "Insufficient Data"),
    name = "Trend Status"
  ) +
  coord_fixed(1.3) +
  labs(
    title    = "Stationarity Assessment of U.S. County Hail Risk (1950-2024)",
    subtitle = paste0("Stationary: ", n_stationary,
                      "  |  Nonstationary Increasing: ",
                      n_increasing,
                      "  |  Nonstationary Decreasing: ",
                      n_decreasing),
    caption  = "Source: NOAA Storm Events Database | Mann-Kendall trend test on annual maximum hail size"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "bottom"
  )

# Part 5: L-moments GEV: stationary counties fitting

# Extract stationary counties (not significant in Mann-Kendall)
stationary_counties <- mk_results |>
  filter(trend == "Not Significant") |>
  mutate(county_id = paste(CZ_NAME, STATE, sep = "_")) |>
  pull(county_id)

# Subset county_list to stationary counties only
county_list_stationary <- county_list[names(county_list) %in% stationary_counties]

# Return level function
return_level <- function(T, location, scale, shape) {
  if (abs(shape) < 1e-6) {
    x_T <- location - scale * log(-log(1 - 1/T))
  } else {
    x_T <- location + (scale / shape) * ((-log(1 - 1/T))^(-shape) - 1)
  }
  return(x_T)
}

gev_stationary <- map_dfr(county_list_stationary, function(df) {
  
  df    <- df |> arrange(YEAR)
  data  <- df$max_hail
  n     <- length(data)
  fips  <- unique(df$CZ_FIPS)[1]
  name  <- unique(df$CZ_NAME)[1]
  state <- unique(df$STATE)[1]
  
  # Skip counties with no variation in data
  if (length(unique(data)) < 2) return(NULL)
  
  # Try L-moments first
  lmom_data    <- samlmu(data)
  gev_fit_lmom <- tryCatch(pelgev(lmom_data), error = function(e) NULL)
  
  # Fall back to MLE if L-moments fail
  if (is.null(gev_fit_lmom)) {
    method_used <- "MLE fallback"
    gev_fit_mle <- tryCatch(
      fevd(data, type = "GEV", method = "MLE"),
      error = function(e) NULL
    )
    if (is.null(gev_fit_mle)) return(NULL)
    location_lmom <- as.numeric(gev_fit_mle$results$par["location"])
    scale_lmom    <- as.numeric(gev_fit_mle$results$par["scale"])
    shape_lmom    <- as.numeric(gev_fit_mle$results$par["shape"])
  } else {
    method_used   <- "L-moments"
    location_lmom <- as.numeric(gev_fit_lmom["xi"])
    scale_lmom    <- as.numeric(gev_fit_lmom["alpha"])
    shape_lmom    <- as.numeric(-gev_fit_lmom["k"])
  }
  
  # Calculate return levels
  rl_50  <- as.numeric(return_level(50,  location_lmom, scale_lmom, shape_lmom))
  rl_100 <- as.numeric(return_level(100, location_lmom, scale_lmom, shape_lmom))
  rl_500 <- as.numeric(return_level(500, location_lmom, scale_lmom, shape_lmom))
  
  data.frame(
    CZ_FIPS       = fips,
    CZ_NAME       = name,
    STATE         = state,
    n_years       = n,
    method_used   = method_used,
    location_lmom = location_lmom,
    scale_lmom    = scale_lmom,
    shape_lmom    = shape_lmom,
    rl_50_lmom    = rl_50,
    rl_100_lmom   = rl_100,
    rl_500_lmom   = rl_500
  )
})

cat("Stationary fitting method breakdown:\n")
print(table(gev_stationary$method_used))

cat("Stationary counties fitted:", nrow(gev_stationary), "\n")

# Distribution of All GEV Parameters

par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))

# Location parameter
hist(gev_stationary$location_lmom,
     breaks = 30,
     main   = "Location Parameter (μ)",
     xlab   = "μ (inches)",
     ylab   = "Frequency",
     col    = "lightblue",
     border = "black")

# Scale parameter
hist(gev_stationary$scale_lmom,
     breaks = 30,
     main   = "Scale Parameter (σ)",
     xlab   = "σ",
     ylab   = "Frequency",
     col    = "lightblue",
     border = "black")

# Shape parameter
hist(gev_stationary$shape_lmom,
     breaks = 30,
     main   = "Shape Parameter (ξ)",
     xlab   = "ξ",
     ylab   = "Frequency",
     col    = "lightblue",
     border = "black")

par(mfrow = c(1, 1))

mtext("Stationary GEV Parameter Distributions (L-moments)", 
      side = 3, line = 0.5, outer = TRUE, cex = 1.1, font = 2)

# Distribution of All Return Levels

par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))

# 50-year return level
hist(gev_stationary$rl_50_lmom,
     breaks = 30,
     main   = "50-Year Return Level",
     xlab   = "Hail Diameter (inches)",
     ylab   = "Frequency",
     col    = "lightgreen",
     border = "black")

# 100-year return level
hist(gev_stationary$rl_100_lmom,
     breaks = 30,
     main   = "100-Year Return Level",
     xlab   = "Hail Diameter (inches)",
     ylab   = "Frequency",
     col    = "lightgreen",
     border = "black")

# 500-year return level
hist(gev_stationary$rl_500_lmom,
     breaks = 30,
     main   = "500-Year Return Level",
     xlab   = "Hail Diameter (inches)",
     ylab   = "Frequency",
     col    = "lightgreen",
     border = "black")

par(mfrow = c(1, 1))

mtext("Stationary GEV Return Level Distributions (L-moments)", 
      side = 3, line = 0.5, outer = TRUE, cex = 1.1, font = 2)

# Summary Table of All Parameters and Return Levels
print(data.frame(
  Statistic     = c("Mean", "Median", "Min", "Max"),
  Location_mu   = c(mean(gev_stationary$location_lmom, na.rm = TRUE),
                    median(gev_stationary$location_lmom, na.rm = TRUE),
                    min(gev_stationary$location_lmom, na.rm = TRUE),
                    max(gev_stationary$location_lmom, na.rm = TRUE)),
  Scale_sigma   = c(mean(gev_stationary$scale_lmom, na.rm = TRUE),
                    median(gev_stationary$scale_lmom, na.rm = TRUE),
                    min(gev_stationary$scale_lmom, na.rm = TRUE),
                    max(gev_stationary$scale_lmom, na.rm = TRUE)),
  Shape_xi      = c(mean(gev_stationary$shape_lmom, na.rm = TRUE),
                    median(gev_stationary$shape_lmom, na.rm = TRUE),
                    min(gev_stationary$shape_lmom, na.rm = TRUE),
                    max(gev_stationary$shape_lmom, na.rm = TRUE)),
  RL_50yr       = c(mean(gev_stationary$rl_50_lmom, na.rm = TRUE),
                    median(gev_stationary$rl_50_lmom, na.rm = TRUE),
                    min(gev_stationary$rl_50_lmom, na.rm = TRUE),
                    max(gev_stationary$rl_50_lmom, na.rm = TRUE)),
  RL_100yr      = c(mean(gev_stationary$rl_100_lmom, na.rm = TRUE),
                    median(gev_stationary$rl_100_lmom, na.rm = TRUE),
                    min(gev_stationary$rl_100_lmom, na.rm = TRUE),
                    max(gev_stationary$rl_100_lmom, na.rm = TRUE)),
  RL_500yr      = c(mean(gev_stationary$rl_500_lmom, na.rm = TRUE),
                    median(gev_stationary$rl_500_lmom, na.rm = TRUE),
                    min(gev_stationary$rl_500_lmom, na.rm = TRUE),
                    max(gev_stationary$rl_500_lmom, na.rm = TRUE))
))

# Part 6: MLE GEV: nonstationary counties fitting

# Extract nonstationary counties (not significant in Mann-Kendall)
nonstationary_counties <- mk_results |>
  filter(trend == "Significant") |>
  mutate(county_id = paste(CZ_NAME, STATE, sep = "_")) |>
  pull(county_id)

# Subset county_list to nonstationary counties only
county_list_nonstationary <- county_list[names(county_list) %in% nonstationary_counties]

gev_nonstationary <- map_dfr(county_list_nonstationary, function(df) {
  
  df <- df |> arrange(YEAR)
  data <- df$max_hail
  n <- length(data)
  fips <<- unique(df$CZ_FIPS)[1]
  name <- unique(df$CZ_NAME)[1]
  state <- unique(df$STATE)[1]
  time <- as.numeric(df$YEAR - min(df$YEAR))
  
  # Try nonstationary MLE first
  gev_fit_nonstat = tryCatch(
    fevd(data,
         data = data.frame(time = time),
         location.fun = ~time,
         type = "GEV",
         method = "MLE"),
    error = function(e) NULL
  )
  
  # Check if nonstationary fit produced valid results
  if (!is.null(gev_fit_nonstat)) {
    mu0   = as.numeric(gev_fit_nonstat$results$par["mu0"])
    mu1   = as.numeric(gev_fit_nonstat$results$par["mu1"])
    sigma = as.numeric(gev_fit_nonstat$results$par["scale"])
    xi    = as.numeric(gev_fit_nonstat$results$par["shape"])
    
    location_end = mu0 + mu1 * max(time)
    rl_50  = as.numeric(return_level(50,  location_end, sigma, xi))
    rl_100 = as.numeric(return_level(100, location_end, sigma, xi))
    rl_500 = as.numeric(return_level(500, location_end, sigma, xi))
    
    # Check physically realistic return levels for all three periods
    if (all(is.finite(c(rl_50, rl_100, rl_500))) &
        all(c(rl_50, rl_100, rl_500) <= 10) &
        all(c(rl_50, rl_100, rl_500) > 0)) {
      return(data.frame(
        CZ_FIPS    = fips,
        CZ_NAME    = name,
        STATE      = state,
        n_years    = n,
        model_used = "Nonstationary MLE",
        mu0        = mu0,
        mu1        = mu1,
        sigma      = sigma,
        xi         = xi,
        rl_50      = rl_50,
        rl_100     = rl_100,
        rl_500     = rl_500
      ))
    }
  }
  
  # Fall back 1: Stationary L-moments
  if (length(unique(data)) < 2) return(NULL)
  
  lmom_data    <- samlmu(data)
  gev_fit_lmom <- tryCatch(pelgev(lmom_data), error = function(e) NULL)
  
  if (!is.null(gev_fit_lmom)) {
    location <- as.numeric(gev_fit_lmom["xi"])
    scale    <- as.numeric(gev_fit_lmom["alpha"])
    shape    <- as.numeric(-gev_fit_lmom["k"])
    
    rl_50  <- as.numeric(return_level(50,  location, scale, shape))
    rl_100 <- as.numeric(return_level(100, location, scale, shape))
    rl_500 <- as.numeric(return_level(500, location, scale, shape))
    
    if (all(is.finite(c(rl_50, rl_100, rl_500))) &
        all(c(rl_50, rl_100, rl_500) <= 10) &
        all(c(rl_50, rl_100, rl_500) > 0)) {
      return(data.frame(
        CZ_FIPS    = fips,
        CZ_NAME    = name,
        STATE      = state,
        n_years    = n,
        model_used = "Stationary L-moments fallback",
        mu0        = NA_real_,
        mu1        = NA_real_,
        sigma      = scale,
        xi         = shape,
        rl_50      = rl_50,
        rl_100     = rl_100,
        rl_500     = rl_500
      ))
    }
  }
  
  # Fall back 2: Stationary MLE
  gev_fit_mle <- tryCatch(
    fevd(data, type = "GEV", method = "MLE"),
    error = function(e) NULL
  )
  if (is.null(gev_fit_mle)) return(NULL)
  
  location <- as.numeric(gev_fit_mle$results$par["location"])
  scale    <- as.numeric(gev_fit_mle$results$par["scale"])
  shape    <- as.numeric(gev_fit_mle$results$par["shape"])
  
  rl_50  <- as.numeric(return_level(50,  location, scale, shape))
  rl_100 <- as.numeric(return_level(100, location, scale, shape))
  rl_500 <- as.numeric(return_level(500, location, scale, shape))
  
  if (any(!is.finite(c(rl_50, rl_100, rl_500)))) return(NULL)
  if (rl_50  > 12) return(NULL)
  if (rl_100 > 12) return(NULL)
  if (rl_500 > 30) return(NULL)
  if (any(c(rl_50, rl_100, rl_500) <= 0)) return(NULL)
  
  data.frame(
    CZ_FIPS    = fips,
    CZ_NAME    = name,
    STATE      = state,
    n_years    = n,
    model_used = "Stationary MLE fallback",
    mu0        = NA_real_,
    mu1        = NA_real_,
    sigma      = scale,
    xi         = shape,
    rl_50      = rl_50,
    rl_100     = rl_100,
    rl_500     = rl_500
  )
})

# Summary of models used
cat("Nonstationary counties fitted:", nrow(gev_nonstationary), "\n")
gev_nonstationary |> count(model_used)

# Summary of trend slopes
summary(gev_nonstationary$mu1)

# Return level summary
print(data.frame(
  Statistic = c("Mean", "Median", "Min", "Max"),
  mu1_slope = c(mean(gev_nonstationary$mu1,   na.rm = TRUE),
                median(gev_nonstationary$mu1, na.rm = TRUE),
                min(gev_nonstationary$mu1,    na.rm = TRUE),
                max(gev_nonstationary$mu1,    na.rm = TRUE)),
  Scale_sigma = c(mean(gev_nonstationary$sigma,   na.rm = TRUE),
                  median(gev_nonstationary$sigma, na.rm = TRUE),
                  min(gev_nonstationary$sigma,    na.rm = TRUE),
                  max(gev_nonstationary$sigma,    na.rm = TRUE)),
  Shape_xi = c(mean(gev_nonstationary$xi,   na.rm = TRUE),
               median(gev_nonstationary$xi, na.rm = TRUE),
               min(gev_nonstationary$xi,    na.rm = TRUE),
               max(gev_nonstationary$xi,    na.rm = TRUE)),
  RL_50yr = c(mean(gev_nonstationary$rl_50,   na.rm = TRUE),
              median(gev_nonstationary$rl_50, na.rm = TRUE),
              min(gev_nonstationary$rl_50,    na.rm = TRUE),
              max(gev_nonstationary$rl_50,    na.rm = TRUE)),
  RL_100yr = c(mean(gev_nonstationary$rl_100,   na.rm = TRUE),
               median(gev_nonstationary$rl_100, na.rm = TRUE),
               min(gev_nonstationary$rl_100,    na.rm = TRUE),
               max(gev_nonstationary$rl_100,    na.rm = TRUE)),
  RL_500yr = c(mean(gev_nonstationary$rl_500,   na.rm = TRUE),
               median(gev_nonstationary$rl_500, na.rm = TRUE),
               min(gev_nonstationary$rl_500,    na.rm = TRUE),
               max(gev_nonstationary$rl_500,    na.rm = TRUE))
))

# Distribution of All Nonstationary GEV Parameters

par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))

# mu0 - intercept
hist(gev_nonstationary$mu0,
     breaks = 30,
     main = "Location Intercept (μ_0)",
     xlab = "μ_0 (inches)",
     ylab = "Frequency",
     col = "lightcoral",
     border = "black")

# mu1 - trend slope
hist(gev_nonstationary$mu1,
     breaks = 30,
     main = "Location Trend Slope (μ_1)",
     xlab = "μ_1 (inches/year)",
     ylab = "Frequency",
     col = "lightcoral",
     border = "black")

# shape
hist(gev_nonstationary$xi,
     breaks = 30,
     main = "Shape Parameter (ξ)",
     xlab = "ξ",
     ylab = "Frequency",
     col = "lightcoral",
     border = "black")

par(mfrow = c(1, 1))

mtext("Nonstationary GEV Parameter Distributions (MLE)", 
      side = 3, line = 0.5, outer = TRUE, cex = 1.1, font = 2)

# Distribution of All Return Levels

par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))

hist(gev_nonstationary$rl_50,
     breaks = 30,
     main = "50-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightsalmon",
     border = "black")

hist(gev_nonstationary$rl_100,
     breaks = 30,
     main = "100-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightsalmon",
     border = "black")

hist(gev_nonstationary$rl_500,
     breaks = 30,
     main = "500-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightsalmon",
     border = "black")

par(mfrow = c(1, 1))

mtext("Nonstationary GEV Return Level Distributions (MLE)", 
      side = 3, line = 0.5, outer = TRUE, cex = 1.1, font = 2)

# 100 Year return-level map

# Get fresh county and state map data
us_counties <- map_data("county")
us_states   <- map_data("state")

# Prepare return level data for joining
rl_map <- bind_rows(
  gev_stationary |>
    mutate(
      state_lower  = tolower(STATE),
      county_lower = clean_county_name(CZ_NAME),
      rl_100_plot  = rl_100_lmom
    ) |>
    select(state_lower, county_lower, rl_100_plot),
  gev_nonstationary |>
    mutate(
      state_lower  = tolower(STATE),
      county_lower = clean_county_name(CZ_NAME),
      rl_100_plot  = rl_100
    ) |>
    select(state_lower, county_lower, rl_100_plot)
) |>
  distinct(state_lower, county_lower, .keep_all = TRUE)

# Join with map data
us_counties <- us_counties |>
  left_join(rl_map,
            by = c("region"    = "state_lower",
                   "subregion" = "county_lower")) |>
  mutate(rl_100_plot = ifelse(is.na(rl_100_plot), NA, rl_100_plot))

# Count shaded counties
n_rl_shaded <- us_counties |>
  filter(!is.na(rl_100_plot)) |>
  distinct(region, subregion) |>
  nrow()

# Plot
ggplot() +
  geom_polygon(data = us_counties,
               aes(x = long, y = lat, group = group,
                   fill = rl_100_plot),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_gradientn(
    colors   = c("lightyellow", "orange", "red", "darkred"),
    na.value = "grey90",
    name     = "100-Year\nReturn Level\n(inches)",
    limits   = c(1, 8),
    oob      = scales::squish
  ) +
  coord_fixed(1.3) +
  labs(
    title    = "100-Year Hail Return Level Across U.S. Counties",
    subtitle = paste0("n = ", n_rl_shaded,
                      " counties | L-Moments (stationary) and MLE (nonstationary)"),
    caption  = "Source: NOAA Storm Events Database (1950-2024)"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "right"
  )

# Part 7: Bootstrap & MCMC on example counties: 1 High-frequency stationary 
# county, high frequency increasing nonstationary county, high-frequency
# nonstationary decreasing county

# Find most data-rich stationary county
top_stationary <- gev_stationary |>
  arrange(desc(n_years)) |>
  select(CZ_NAME, STATE, n_years, rl_100_lmom) |>
  head(1)

top_stationary

# Find county with increasing trend (most frequency)
top_increasing <- gev_nonstationary |>
  filter(mu1 > 0) |>
  left_join(annual_max |> 
              group_by(CZ_NAME, STATE) |> 
              summarise(data_years = n_distinct(YEAR), .groups = "drop"),
            by = c("CZ_NAME", "STATE")) |>
  arrange(desc(data_years)) |>
  select(CZ_NAME, STATE, n_years, mu1, rl_100) |>
  head(1)

top_increasing

# Find county with decreasing trend (most frequency)
top_decreasing <- gev_nonstationary |>
  filter(mu1 < 0) |>
  left_join(annual_max |> 
              group_by(CZ_NAME, STATE) |> 
              summarise(data_years = n_distinct(YEAR), .groups = "drop"),
            by = c("CZ_NAME", "STATE")) |>
  arrange(desc(data_years)) |>
  select(CZ_NAME, STATE, n_years, mu1, rl_100) |>
  head(1)

top_decreasing

# Extract example counties from annual_max
example_stationary <- annual_max |>
  filter(CZ_NAME == "OKLAHOMA" & STATE == "OKLAHOMA") |>
  arrange(YEAR)

example_increasing <- annual_max |>
  filter(CZ_NAME == "PENNINGTON" & STATE == "SOUTH DAKOTA") |>
  arrange(YEAR)

example_decreasing <- annual_max |>
  filter(CZ_NAME == "JEFFERSON" & STATE == "ALABAMA") |>
  arrange(YEAR)

# Bootstrap settings
n_boot <- 1000
return_periods <- c(50, 100, 500)

# Bootstrap - Stationary County (Oklahoma County, OK)

data_stat <- example_stationary$max_hail
n_stat <- length(data_stat)

# Run bootstrap
boot_stat <- replicate(n_boot, {
  
  # Resample with replacement
  boot_sample <- sample(data_stat, size = n_stat, replace = TRUE)
  
  # Fit GEV using L-Moments
  lmom_data <- samlmu(boot_sample)
  gev_fit <- pelgev(lmom_data)
  
  location <- as.numeric(gev_fit["xi"])
  scale <- as.numeric(gev_fit["alpha"])
  shape <- as.numeric(-gev_fit["k"])
  
  # Return levels
  rl_50 = return_level(50,  location, scale, shape)
  rl_100 = return_level(100, location, scale, shape)
  rl_500 = return_level(500, location, scale, shape)
  
  if (rl_50  > 12) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  if (rl_100 > 12) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  if (rl_500 > 30) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  
  c(rl_50  = rl_50,
    rl_100 = rl_100,
    rl_500 = rl_500)
})

# Extract confidence intervals
boot_stat_ci <- data.frame(
  Return_Period = c("50-Year", "100-Year", "500-Year"),
  Point_Est = c(
    return_level(50,  as.numeric(pelgev(samlmu(data_stat))["xi"]),
                 as.numeric(pelgev(samlmu(data_stat))["alpha"]),
                 as.numeric(-pelgev(samlmu(data_stat))["k"])),
    return_level(100, as.numeric(pelgev(samlmu(data_stat))["xi"]),
                 as.numeric(pelgev(samlmu(data_stat))["alpha"]),
                 as.numeric(-pelgev(samlmu(data_stat))["k"])),
    return_level(500, as.numeric(pelgev(samlmu(data_stat))["xi"]),
                 as.numeric(pelgev(samlmu(data_stat))["alpha"]),
                 as.numeric(-pelgev(samlmu(data_stat))["k"]))
  ),
  Lower_95 = apply(boot_stat, 1, quantile, 0.025, na.rm = TRUE),
  Upper_95 = apply(boot_stat, 1, quantile, 0.975, na.rm = TRUE)
)

print(boot_stat_ci)

# Triplot - Bootstrap distributions for 50, 100, and 500-year return levels
# Oklahoma County, OK (Stationary)
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))

# 50-year
hist(boot_stat["rl_50",],
     breaks = 30,
     main = "50-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightblue",
     border = "black")
abline(v = boot_stat_ci$Point_Est[1], col = "red",  lwd = 2)
abline(v = boot_stat_ci$Lower_95[1],  col = "blue", lwd = 2, lty = 2)
abline(v = boot_stat_ci$Upper_95[1],  col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

# 100-year
hist(boot_stat["rl_100",],
     breaks = 30,
     main = "100-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightblue",
     border = "black")
abline(v = boot_stat_ci$Point_Est[2], col = "red",  lwd = 2)
abline(v = boot_stat_ci$Lower_95[2],  col = "blue", lwd = 2, lty = 2)
abline(v = boot_stat_ci$Upper_95[2],  col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

# 500-year
hist(boot_stat["rl_500",],
     breaks = 30,
     main = "500-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightblue",
     border = "black")
abline(v = boot_stat_ci$Point_Est[3], col = "red",  lwd = 2)
abline(v = boot_stat_ci$Lower_95[3],  col = "blue", lwd = 2, lty = 2)
abline(v = boot_stat_ci$Upper_95[3],  col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

mtext("Bootstrap Return Level Distributions - Oklahoma County, OK (Stationary)", 
      side = 3, line = -0.5, outer = TRUE, cex = 1.1, font = 2)

par(mfrow = c(1, 1))

# Bootstrap - Nonstationary Increasing (Pennington County, SD)

data_inc <- example_increasing$max_hail
time_inc <- as.numeric(example_increasing$YEAR - min(example_increasing$YEAR))
n_inc <- length(data_inc)

# Run bootstrap
boot_inc <- replicate(n_boot, {
  
  # Resample with replacement (keep time index paired with data)
  boot_idx <- sample(1:n_inc, size = n_inc, replace = TRUE)
  boot_sample <- data_inc[boot_idx]
  boot_time <- time_inc[boot_idx]
  
  # Fit nonstationary GEV via MLE
  lmom_data = samlmu(data_inc)
  lmom_fit = pelgev(lmom_data)
  
  fit <- fevd(boot_sample,
         data = data.frame(time = boot_time),
         location.fun = ~time,
         type = "GEV",
         method = "MLE",
         initial = list(mu0 = as.numeric(lmom_fit["xi"]),
                        mu1 = 0,
                        scale = as.numeric(lmom_fit["alpha"]),
                        shape = as.numeric(-lmom_fit["k"])))
  
  
  mu0 = as.numeric(fit$results$par["mu0"])
  mu1 = as.numeric(fit$results$par["mu1"])
  sigma = as.numeric(fit$results$par["scale"])
  xi = as.numeric(fit$results$par["shape"])
  
  # Return levels at end of record
  location_end = mu0 + mu1 * max(time_inc)
  rl_50  = return_level(50,  location_end, sigma, xi)
  rl_100 = return_level(100, location_end, sigma, xi)
  rl_500 = return_level(500, location_end, sigma, xi)
  
  # Cap at 12 inches to maintain consistency and reasonability
  if (rl_50  > 12) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  if (rl_100 > 12) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  if (rl_500 > 30) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  
  c(rl_50  = rl_50,
    rl_100 = rl_100,
    rl_500 = rl_500)
  
})

# Extract confidence intervals
boot_inc_ci <- data.frame(
  Return_Period = c("50-Year", "100-Year", "500-Year"),
  Lower_95 = apply(boot_inc, 1, quantile, 0.025, na.rm = TRUE),
  Upper_95 = apply(boot_inc, 1, quantile, 0.975, na.rm = TRUE)
)

print(boot_inc_ci)

# Get point estimate for increasing county - 50 year
point_est_inc_50 <- gev_nonstationary |>
  filter(CZ_NAME == top_increasing$CZ_NAME & 
           STATE == top_increasing$STATE) |>
  pull(rl_50)

# Get point estimate for increasing county - 100 year
point_est_inc_100 <- gev_nonstationary |>
  filter(CZ_NAME == top_increasing$CZ_NAME & 
           STATE == top_increasing$STATE) |>
  pull(rl_100)

# Get point estimate for increasing county - 500 year
point_est_inc_500 <- gev_nonstationary |>
  filter(CZ_NAME == top_increasing$CZ_NAME & 
           STATE == top_increasing$STATE) |>
  pull(rl_500)

# Triplot - Bootstrap distributions for 50, 100, and 500-year return levels
# Pennington County, SD (Nonstationary Increasing)
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))

# 50-year
hist(boot_inc["rl_50",],
     breaks = 30,
     main = "50-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightgreen",
     border = "black")
abline(v = point_est_inc_50,        col = "red",  lwd = 2)
abline(v = boot_inc_ci$Lower_95[1], col = "blue", lwd = 2, lty = 2)
abline(v = boot_inc_ci$Upper_95[1], col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

# 100-year
hist(boot_inc["rl_100",],
     breaks = 30,
     main = "100-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightgreen",
     border = "black")
abline(v = point_est_inc_100,       col = "red",  lwd = 2)
abline(v = boot_inc_ci$Lower_95[2], col = "blue", lwd = 2, lty = 2)
abline(v = boot_inc_ci$Upper_95[2], col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

# 500-year
hist(boot_inc["rl_500",],
     breaks = 30,
     main = "500-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightgreen",
     border = "black")
abline(v = point_est_inc_500,       col = "red",  lwd = 2)
abline(v = boot_inc_ci$Lower_95[3], col = "blue", lwd = 2, lty = 2)
abline(v = boot_inc_ci$Upper_95[3], col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

mtext("Bootstrap Return Level Distributions - Pennington County, SD (Nonstationary Increasing)", 
      side = 3, line = -0.5, outer = TRUE, cex = 1.1, font = 2)

par(mfrow = c(1, 1))

# Bootstrap - Nonstationary Decreasing (Jefferson County, AL)

data_dec <- example_decreasing$max_hail
time_dec <- as.numeric(example_decreasing$YEAR - min(example_decreasing$YEAR))
n_dec <- length(data_dec)

# Run bootstrap
boot_dec <- replicate(n_boot, {
  
  boot_idx <- sample(1:n_dec, size = n_dec, replace = TRUE)
  boot_sample <- data_dec[boot_idx]
  boot_time <- time_dec[boot_idx]
  
  lmom_data = samlmu(data_dec)
  lmom_fit = pelgev(lmom_data)
  
  fit <- fevd(boot_sample,
         data = data.frame(time = boot_time),
         location.fun = ~time,
         type = "GEV",
         method = "MLE",
         initial = list(mu0 = as.numeric(lmom_fit["xi"]),
                        mu1 = 0,
                        scale = as.numeric(lmom_fit["alpha"]),
                        shape = as.numeric(-lmom_fit["k"])))
  
  mu0 = as.numeric(fit$results$par["mu0"])
  mu1 = as.numeric(fit$results$par["mu1"])
  sigma = as.numeric(fit$results$par["scale"])
  xi = as.numeric(fit$results$par["shape"])
  
  location_end = mu0 + mu1 * max(time_dec)
  rl_50  = return_level(50,  location_end, sigma, xi)
  rl_100 = return_level(100, location_end, sigma, xi)
  rl_500 = return_level(500, location_end, sigma, xi)
  
  if (rl_50  > 12) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  if (rl_100 > 12) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  if (rl_500 > 30) return(c(rl_50 = NA, rl_100 = NA, rl_500 = NA))
  
  c(rl_50  = rl_50,
    rl_100 = rl_100,
    rl_500 = rl_500)
})

# Extract confidence intervals
boot_dec_ci <- data.frame(
  Return_Period = c("50-Year", "100-Year", "500-Year"),
  Lower_95 = apply(boot_dec, 1, quantile, 0.025, na.rm = TRUE),
  Upper_95 = apply(boot_dec, 1, quantile, 0.975, na.rm = TRUE)
)

print(boot_dec_ci)

# Get point estimate for decreasing county - 50 year
point_est_dec_50 <- gev_nonstationary |>
  filter(CZ_NAME == top_decreasing$CZ_NAME & 
           STATE == top_decreasing$STATE) |>
  pull(rl_50)

# Get point estimate for decreasing county - 100 year
point_est_dec_100 <- gev_nonstationary |>
  filter(CZ_NAME == top_decreasing$CZ_NAME & 
           STATE == top_decreasing$STATE) |>
  pull(rl_100)

# Get point estimate for decreasing county - 500 year
point_est_dec_500 <- gev_nonstationary |>
  filter(CZ_NAME == top_decreasing$CZ_NAME & 
           STATE == top_decreasing$STATE) |>
  pull(rl_500)

# Triplot - Bootstrap distributions for 50, 100, and 500-year return levels
# Jefferson County, AL (Nonstationary Decreasing)
par(mfrow = c(1, 3), oma = c(0, 0, 2, 0))

# 50-year
hist(boot_dec["rl_50",],
     breaks = 30,
     main = "50-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightsalmon",
     border = "black")
abline(v = point_est_dec_50,        col = "red",  lwd = 2)
abline(v = boot_dec_ci$Lower_95[1], col = "blue", lwd = 2, lty = 2)
abline(v = boot_dec_ci$Upper_95[1], col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

# 100-year
hist(boot_dec["rl_100",],
     breaks = 30,
     main = "100-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightsalmon",
     border = "black")
abline(v = point_est_dec_100,       col = "red",  lwd = 2)
abline(v = boot_dec_ci$Lower_95[2], col = "blue", lwd = 2, lty = 2)
abline(v = boot_dec_ci$Upper_95[2], col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

# 500-year
hist(boot_dec["rl_500",],
     breaks = 30,
     main = "500-Year Return Level",
     xlab = "Hail Diameter (inches)",
     ylab = "Frequency",
     col = "lightsalmon",
     border = "black")
abline(v = point_est_dec_500,       col = "red",  lwd = 2)
abline(v = boot_dec_ci$Lower_95[3], col = "blue", lwd = 2, lty = 2)
abline(v = boot_dec_ci$Upper_95[3], col = "blue", lwd = 2, lty = 2)
legend("topright",
       legend = c("Point Estimate", "95% CI"),
       col    = c("red", "blue"),
       lwd    = c(2, 2),
       lty    = c(1, 2))

mtext("Bootstrap Return Level Distributions - Jefferson County, AL (Nonstationary Decreasing)", 
      side = 3, line = -0.5, outer = TRUE, cex = 1.1, font = 2)

par(mfrow = c(1, 1))

# Summary Table - Bootstrap CI Comparison Across Counties
data.frame(
  County        = c("Oklahoma Co., OK", "Pennington Co., SD", "Jefferson Co., AL"),
  Type          = c("Stationary", "Nonstationary Increasing", "Nonstationary Decreasing"),
  # 50-year
  RL_50_Est     = c(boot_stat_ci$Point_Est[1], point_est_inc_50,        point_est_dec_50),
  RL_50_Lower   = c(boot_stat_ci$Lower_95[1],  boot_inc_ci$Lower_95[1], boot_dec_ci$Lower_95[1]),
  RL_50_Upper   = c(boot_stat_ci$Upper_95[1],  boot_inc_ci$Upper_95[1], boot_dec_ci$Upper_95[1]),
  RL_50_CI_Width = c(boot_stat_ci$Upper_95[1] - boot_stat_ci$Lower_95[1],
                     boot_inc_ci$Upper_95[1]  - boot_inc_ci$Lower_95[1],
                     boot_dec_ci$Upper_95[1]  - boot_dec_ci$Lower_95[1]),
  # 100-year
  RL_100_Est    = c(boot_stat_ci$Point_Est[2], point_est_inc_100,       point_est_dec_100),
  RL_100_Lower  = c(boot_stat_ci$Lower_95[2],  boot_inc_ci$Lower_95[2], boot_dec_ci$Lower_95[2]),
  RL_100_Upper  = c(boot_stat_ci$Upper_95[2],  boot_inc_ci$Upper_95[2], boot_dec_ci$Upper_95[2]),
  RL_100_CI_Width = c(boot_stat_ci$Upper_95[2] - boot_stat_ci$Lower_95[2],
                      boot_inc_ci$Upper_95[2]  - boot_inc_ci$Lower_95[2],
                      boot_dec_ci$Upper_95[2]  - boot_dec_ci$Lower_95[2]),
  # 500-year
  RL_500_Est    = c(boot_stat_ci$Point_Est[3], point_est_inc_500,       point_est_dec_500),
  RL_500_Lower  = c(boot_stat_ci$Lower_95[3],  boot_inc_ci$Lower_95[3], boot_dec_ci$Lower_95[3]),
  RL_500_Upper  = c(boot_stat_ci$Upper_95[3],  boot_inc_ci$Upper_95[3], boot_dec_ci$Upper_95[3]),
  RL_500_CI_Width = c(boot_stat_ci$Upper_95[3] - boot_stat_ci$Lower_95[3],
                      boot_inc_ci$Upper_95[3]  - boot_inc_ci$Lower_95[3],
                      boot_dec_ci$Upper_95[3]  - boot_dec_ci$Lower_95[3])
)

# Part 7 continued: MCMC for example counties

return_periods <- c(50, 100, 500)

# MCMC - Stationary County (Oklahoma County, OK)

data_stat = example_stationary$max_hail

# Set flat prior and draw posterior samples
prior_gev = set_prior(prior = "flat", model = "gev")

gev_bayes_stat = rpost(
  n     = 20000,
  model = "gev",
  data  = data_stat,
  prior = prior_gev
)

cat("\n=== Bayesian Posterior Summary - Oklahoma County, OK ===\n")
print(summary(gev_bayes_stat))

# Extract posterior samples
post_stat = gev_bayes_stat$sim_vals

location_post_stat = post_stat[, "mu"]
scale_post_stat    = post_stat[, "sigma"]
shape_post_stat    = post_stat[, "xi"]

# Compute return levels from posterior
rl_mcmc_stat = matrix(NA_real_, nrow = length(location_post_stat),
                      ncol = length(return_periods))

for (i in 1:length(location_post_stat)) {
  for (j in seq_along(return_periods)) {
    rl_mcmc_stat[i, j] = return_level(
      return_periods[j],
      location_post_stat[i],
      scale_post_stat[i],
      shape_post_stat[i]
    )
  }
}

# Credible intervals
rl_point_stat  = apply(rl_mcmc_stat, 2, median,   na.rm = TRUE)
ci_lower_stat  = apply(rl_mcmc_stat, 2, quantile, 0.025, na.rm = TRUE)
ci_upper_stat  = apply(rl_mcmc_stat, 2, quantile, 0.975, na.rm = TRUE)

cat("\n=== Bayesian Return Level Credible Intervals - Oklahoma County, OK ===\n")
data.frame(
  Return_Period  = return_periods,
  Point_Estimate = rl_point_stat,
  Lower_CI       = ci_lower_stat,
  Upper_CI       = ci_upper_stat
)

# Return level plot with credible band
T_smooth = 10^seq(log10(1.001), log10(1000), length.out = 400)

rl_smooth_draws_stat = matrix(NA_real_, nrow = length(location_post_stat),
                              ncol = length(T_smooth))
for (i in 1:length(location_post_stat)) {
  rl_smooth_draws_stat[i, ] = sapply(T_smooth, return_level,
                                     location_post_stat[i],
                                     scale_post_stat[i],
                                     shape_post_stat[i])
}

rl_smooth_stat  = apply(rl_smooth_draws_stat, 2, median)
rl_lower_smooth_stat = apply(rl_smooth_draws_stat, 2, quantile, 0.025)
rl_upper_smooth_stat = apply(rl_smooth_draws_stat, 2, quantile, 0.975)

# Empirical plotting positions
n_stat = length(data_stat)
rank_stat = 1:n_stat
T_emp_stat = (n_stat + 1) / (n_stat + 1 - rank_stat)

plot(T_smooth, rl_smooth_stat,
     log  = "x",
     type = "l",
     lwd  = 3,
     col  = "darkgreen",
     xlab = "Return Period (years)",
     ylab = "Hail Diameter (inches)",
     main = "MCMC Return Level Plot - Oklahoma County, OK (Stationary)",
     ylim = c(min(rl_lower_smooth_stat, na.rm = TRUE) - 0.5,
              max(rl_upper_smooth_stat, na.rm = TRUE) + 0.5))

polygon(c(T_smooth, rev(T_smooth)),
        c(rl_lower_smooth_stat, rev(rl_upper_smooth_stat)),
        col    = rgb(0, 0.6, 0, 0.2),
        border = NA)

points(T_emp_stat, sort(data_stat),
       pch = 19,
       col = "darkblue")

legend("bottomright",
       legend = c("Bayesian Median", "95% Credible Interval", "Empirical Data"),
       col    = c("darkgreen", rgb(0, 0.6, 0, 0.2), "darkblue"),
       lty    = c(1, NA, NA),
       lwd    = c(3, NA, NA),
       pch    = c(NA, 15, 19),
       pt.cex = c(1, 2, 1),
       bty    = "n")

# MCMC - Nonstationary Increasing (Pennington County, SD)

data_inc = example_increasing$max_hail

gev_bayes_inc = rpost(
  n     = 20000,
  model = "gev",
  data  = data_inc,
  prior = prior_gev
)

cat("\n=== Bayesian Posterior Summary - Pennington County, SD ===\n")
print(summary(gev_bayes_inc))

# Extract posterior samples
post_inc = gev_bayes_inc$sim_vals

location_post_inc = post_inc[, "mu"]
scale_post_inc    = post_inc[, "sigma"]
shape_post_inc    = post_inc[, "xi"]

# Compute return levels from posterior
rl_mcmc_inc = matrix(NA_real_, nrow = length(location_post_inc),
                     ncol = length(return_periods))

for (i in 1:length(location_post_inc)) {
  for (j in seq_along(return_periods)) {
    rl_mcmc_inc[i, j] = return_level(
      return_periods[j],
      location_post_inc[i],
      scale_post_inc[i],
      shape_post_inc[i]
    )
  }
}

# Credible intervals
rl_point_inc = apply(rl_mcmc_inc, 2, median,   na.rm = TRUE)
ci_lower_inc = apply(rl_mcmc_inc, 2, quantile, 0.025, na.rm = TRUE)
ci_upper_inc = apply(rl_mcmc_inc, 2, quantile, 0.975, na.rm = TRUE)

cat("\n=== Bayesian Return Level Credible Intervals - Pennington County, SD ===\n")
data.frame(
  Return_Period  = return_periods,
  Point_Estimate = rl_point_inc,
  Lower_CI       = ci_lower_inc,
  Upper_CI       = ci_upper_inc
)

# Return level plot with credible band
rl_smooth_draws_inc = matrix(NA_real_, nrow = length(location_post_inc),
                             ncol = length(T_smooth))
for (i in 1:length(location_post_inc)) {
  rl_smooth_draws_inc[i, ] = sapply(T_smooth, return_level,
                                    location_post_inc[i],
                                    scale_post_inc[i],
                                    shape_post_inc[i])
}

rl_smooth_inc  = apply(rl_smooth_draws_inc, 2, median)
rl_lower_smooth_inc = apply(rl_smooth_draws_inc, 2, quantile, 0.025)
rl_upper_smooth_inc = apply(rl_smooth_draws_inc, 2, quantile, 0.975)

n_inc = length(data_inc)
rank_inc = 1:n_inc
T_emp_inc = (n_inc + 1) / (n_inc + 1 - rank_inc)

plot(T_smooth, rl_smooth_inc,
     log  = "x",
     type = "l",
     lwd  = 3,
     col  = "darkgreen",
     xlab = "Return Period (years)",
     ylab = "Hail Diameter (inches)",
     main = "MCMC Return Level Plot - Pennington County, SD (Nonstationary Increasing)",
     ylim = c(min(rl_lower_smooth_inc, na.rm = TRUE) - 0.5,
              max(rl_upper_smooth_inc, na.rm = TRUE) + 0.5))

polygon(c(T_smooth, rev(T_smooth)),
        c(rl_lower_smooth_inc, rev(rl_upper_smooth_inc)),
        col    = rgb(0, 0.6, 0, 0.2),
        border = NA)

points(T_emp_inc, sort(data_inc),
       pch = 19,
       col = "darkblue")

legend("bottomright",
       legend = c("Bayesian Median", "95% Credible Interval", "Empirical Data"),
       col    = c("darkgreen", rgb(0, 0.6, 0, 0.2), "darkblue"),
       lty    = c(1, NA, NA),
       lwd    = c(3, NA, NA),
       pch    = c(NA, 15, 19),
       pt.cex = c(1, 2, 1),
       bty    = "n")

# MCMC - Nonstationary Decreasing (Jefferson County, AL)

data_dec = example_decreasing$max_hail

gev_bayes_dec = rpost(
  n     = 20000,
  model = "gev",
  data  = data_dec,
  prior = prior_gev
)

cat("\n=== Bayesian Posterior Summary - Jefferson County, AL ===\n")
print(summary(gev_bayes_dec))

# Extract posterior samples
post_dec = gev_bayes_dec$sim_vals

location_post_dec = post_dec[, "mu"]
scale_post_dec    = post_dec[, "sigma"]
shape_post_dec    = post_dec[, "xi"]

# Compute return levels from posterior
rl_mcmc_dec = matrix(NA_real_, nrow = length(location_post_dec),
                     ncol = length(return_periods))

for (i in 1:length(location_post_dec)) {
  for (j in seq_along(return_periods)) {
    rl_mcmc_dec[i, j] = return_level(
      return_periods[j],
      location_post_dec[i],
      scale_post_dec[i],
      shape_post_dec[i]
    )
  }
}

# Credible intervals
rl_point_dec = apply(rl_mcmc_dec, 2, median,   na.rm = TRUE)
ci_lower_dec = apply(rl_mcmc_dec, 2, quantile, 0.025, na.rm = TRUE)
ci_upper_dec = apply(rl_mcmc_dec, 2, quantile, 0.975, na.rm = TRUE)

cat("\n=== Bayesian Return Level Credible Intervals - Jefferson County, AL ===\n")
data.frame(
  Return_Period  = return_periods,
  Point_Estimate = rl_point_dec,
  Lower_CI       = ci_lower_dec,
  Upper_CI       = ci_upper_dec
)

# Return level plot with credible band
rl_smooth_draws_dec = matrix(NA_real_, nrow = length(location_post_dec),
                             ncol = length(T_smooth))
for (i in 1:length(location_post_dec)) {
  rl_smooth_draws_dec[i, ] = sapply(T_smooth, return_level,
                                    location_post_dec[i],
                                    scale_post_dec[i],
                                    shape_post_dec[i])
}

rl_smooth_dec  = apply(rl_smooth_draws_dec, 2, median)
rl_lower_smooth_dec = apply(rl_smooth_draws_dec, 2, quantile, 0.025)
rl_upper_smooth_dec = apply(rl_smooth_draws_dec, 2, quantile, 0.975)

n_dec = length(data_dec)
rank_dec = 1:n_dec
T_emp_dec = (n_dec + 1) / (n_dec + 1 - rank_dec)

plot(T_smooth, rl_smooth_dec,
     log  = "x",
     type = "l",
     lwd  = 3,
     col  = "darkgreen",
     xlab = "Return Period (years)",
     ylab = "Hail Diameter (inches)",
     main = "MCMC Return Level Plot - Jefferson County, AL (Nonstationary Decreasing)",
     ylim = c(min(rl_lower_smooth_dec, na.rm = TRUE) - 0.5,
              max(rl_upper_smooth_dec, na.rm = TRUE) + 0.5))

polygon(c(T_smooth, rev(T_smooth)),
        c(rl_lower_smooth_dec, rev(rl_upper_smooth_dec)),
        col    = rgb(0, 0.6, 0, 0.2),
        border = NA)

points(T_emp_dec, sort(data_dec),
       pch = 19,
       col = "darkblue")

legend("bottomright",
       legend = c("Bayesian Median", "95% Credible Interval", "Empirical Data"),
       col    = c("darkgreen", rgb(0, 0.6, 0, 0.2), "darkblue"),
       lty    = c(1, NA, NA),
       lwd    = c(3, NA, NA),
       pch    = c(NA, 15, 19),
       pt.cex = c(1, 2, 1),
       bty    = "n")

# Summary Table - MCMC CI Comparison Across Counties

data.frame(
  County         = c("Oklahoma Co., OK", "Pennington Co., SD", "Jefferson Co., AL"),
  Type           = c("Stationary", "Nonstationary Increasing", "Nonstationary Decreasing"),
  RL_50_Est      = c(rl_point_stat[1], rl_point_inc[1], rl_point_dec[1]),
  RL_50_Lower    = c(ci_lower_stat[1], ci_lower_inc[1], ci_lower_dec[1]),
  RL_50_Upper    = c(ci_upper_stat[1], ci_upper_inc[1], ci_upper_dec[1]),
  RL_100_Est     = c(rl_point_stat[2], rl_point_inc[2], rl_point_dec[2]),
  RL_100_Lower   = c(ci_lower_stat[2], ci_lower_inc[2], ci_lower_dec[2]),
  RL_100_Upper   = c(ci_upper_stat[2], ci_upper_inc[2], ci_upper_dec[2]),
  RL_500_Est     = c(rl_point_stat[3], rl_point_inc[3], rl_point_dec[3]),
  RL_500_Lower   = c(ci_lower_stat[3], ci_lower_inc[3], ci_lower_dec[3]),
  RL_500_Upper   = c(ci_upper_stat[3], ci_upper_inc[3], ci_upper_dec[3])
)

# Part 8: Exposure & Vulnerability

# Load NRI Exposure Data
nri <- read_csv("C:/Users/14046/OneDrive/Documents/R scripts Weather Risk/National_Risk_Index_Counties.csv") |>
  select(
    CZ_FIPS = `State-County FIPS Code`,
    county = `County Name`,
    state = `State Name Abbreviation`,
    bldg_exposure = `Hail - Exposure - Building Value`,
    ag_exposure   = `Hail - Exposure - Agriculture Value`,
    loss_ratio_bldg = `Hail - Historic Loss Ratio - Buildings`,
    loss_ratio_ag = `Hail - Historic Loss Ratio - Agriculture`,
    fema_eal_bldg = `Hail - Expected Annual Loss - Building Value`,
    fema_eal_ag = `Hail - Expected Annual Loss - Agriculture Value`
  ) |>
  mutate(CZ_FIPS = as.integer(CZ_FIPS))

# Clean NRI for joining (ensure matching names with NOAA)
nri_clean <- nri |>
  mutate(
    state_full   = toupper(state.name[match(state, state.abb)]),
    county_upper = toupper(county),
    county_clean = str_remove(county_upper,
                              " COUNTY| PARISH| BOROUGH| CENSUS AREA| MUNICIPALITY| CITY"),
    county_clean = str_trim(county_clean),
    county_id    = paste(county_clean, state_full, sep = "_")
  )

# Remove duplicates - keep first occurrence (county over city)
nri_clean <- nri_clean |>
  group_by(county_id) |>
  slice(1) |>
  ungroup()

# Basic summary
cat("NRI counties loaded:", nrow(nri), "\n")
summary(nri$bldg_exposure)
summary(nri$ag_exposure)

# Derive Diameter-Specific Damage Ratios from NOAA Data
# Rather than using FEMA single average loss ratio, we derive bin-specific
# damage ratios from observed NOAA storm event damage reports

damage_ratios <- data.frame(
  bin = c("1.0-2.0 in", "2.0-3.0 in", "3.0-4.0 in", "4.0+ in"),
  min_size = c(1.0, 2.0, 3.0, 4.0),
  max_size = c(2.0, 3.0, 4.0, Inf),
  prop_damage_mean = c(
    mean(hail_data_1_2$DAMAGE_PROPERTY,  na.rm = TRUE),
    mean(hail_data_2_3$DAMAGE_PROPERTY,  na.rm = TRUE),
    mean(hail_data_3_4$DAMAGE_PROPERTY,  na.rm = TRUE),
    mean(hail_data_4Plus$DAMAGE_PROPERTY, na.rm = TRUE)
  ),
  ag_damage_mean   = c(
    mean(hail_data_1_2$DAMAGE_CROPS,  na.rm = TRUE),
    mean(hail_data_2_3$DAMAGE_CROPS,  na.rm = TRUE),
    mean(hail_data_3_4$DAMAGE_CROPS,  na.rm = TRUE),
    mean(hail_data_4Plus$DAMAGE_CROPS, na.rm = TRUE)
  )
)

print(damage_ratios)

# Classify GEV Return Levels into Hail Size Bins
# Assign each county's 100-year return level to a damage bin

classify_bin <- function(rl) {
  case_when(
    rl >= 0.99 & rl < 1.99 ~ "1.0-2.0 in",
    rl >= 1.99 & rl < 2.99 ~ "2.0-3.0 in",
    rl >= 2.99 & rl < 3.99 ~ "3.0-4.0 in",
    rl >= 3.99              ~ "4.0+ in",
    TRUE                    ~ NA_character_
  )
}

# Apply to stationary counties
gev_stationary <- gev_stationary |>
  mutate(
    bin_50  = classify_bin(rl_50_lmom),
    bin_100 = classify_bin(rl_100_lmom),
    bin_500 = classify_bin(rl_500_lmom)
  )

# Apply to nonstationary counties
gev_nonstationary <- gev_nonstationary |>
  mutate(
    bin_50  = classify_bin(rl_50),
    bin_100 = classify_bin(rl_100),
    bin_500 = classify_bin(rl_500)
  )

# Join GEV Results with NRI Exposure
# Combine stationary and nonstationary into one results dataframe
gev_all <- bind_rows(
  gev_stationary |>
    mutate(
      model  = "Stationary",
      rl_50  = rl_50_lmom,
      rl_100 = rl_100_lmom,
      rl_500 = rl_500_lmom
    ) |>
    select(CZ_FIPS, CZ_NAME, STATE, n_years, model,
           rl_50, rl_100, rl_500, bin_50, bin_100, bin_500),
  gev_nonstationary |>
    mutate(model = case_when(
      model_used == "Nonstationary MLE" & mu1 > 0  ~ "Nonstationary Increasing",
      model_used == "Nonstationary MLE" & mu1 <= 0 ~ "Nonstationary Decreasing",
      TRUE ~ "Nonstationary (Stationary Fallback)"
    )) |>
    select(CZ_FIPS, CZ_NAME, STATE, n_years, model,
           rl_50, rl_100, rl_500, bin_50, bin_100, bin_500)
) |>
  mutate(county_id = paste(CZ_NAME, STATE, sep = "_"))

gev_all <- gev_all |>
  mutate(
    CZ_NAME = case_when(
      # --- Specific State/County Logic ---
      # --- Connecticut Mapping (Old County -> New Planning Region) ---
      # This mapping is not perfect since the old dataset uses counties while
      # the new data set using planning regions 
      # (counties were abolished by connecticut in 2022)
      # Connecticut old county to NRI planning region
      STATE == "CONNECTICUT" & CZ_NAME == "NEW LONDON" ~ "SOUTHEASTERN CONNECTICUT",
      STATE == "CONNECTICUT" & CZ_NAME == "WINDHAM"    ~ "NORTHEASTERN CONNECTICUT",
      STATE == "CONNECTICUT" & CZ_NAME == "HARTFORD"   ~ "CAPITOL",
      STATE == "CONNECTICUT" & CZ_NAME == "TOLLAND"    ~ "CAPITOL",
      STATE == "CONNECTICUT" & CZ_NAME == "LITCHFIELD" ~ "NORTHWEST HILLS",
      STATE == "CONNECTICUT" & CZ_NAME == "MIDDLESEX"  ~ "LOWER CONNECTICUT RIVER VALLEY",
      STATE == "CONNECTICUT" & CZ_NAME == "NEW HAVEN"  ~ "SOUTH CENTRAL CONNECTICUT",
      STATE == "CONNECTICUT" & CZ_NAME == "NEW HAVEN"  ~ "NAUGATUCK",
      STATE == "CONNECTICUT" & CZ_NAME == "FAIRFIELD"  ~ "WESTERN CONNECTICUT",
      STATE == "CONNECTICUT" & CZ_NAME == "FAIRFIELD"  ~ "GREATER BRIDGEPORT",
    
    # LA SALLE: Ensure both IL and LA are standardized (usually to "LA SALLE")
    STATE == "ILLINOIS"  & CZ_NAME == "LA SALLE" ~ "LASALLE",
    STATE == "LOUISIANA" & CZ_NAME == "LA SALLE" ~ "LASALLE",
    
    # DESOTO: Mississippi is usually "DESOTO" or "DE SOTO"
    STATE == "MISSISSIPPI" & CZ_NAME == "DE SOTO" ~ "DESOTO",
    
    # DEWITT: Texas is usually "DEWITT", Illinois is "DE WITT"
    STATE == "TEXAS"    & CZ_NAME == "DE WITT" ~ "DEWITT",
    STATE == "ILLINOIS" & CZ_NAME == "DEWITT"  ~ "DE WITT",
    STATE == "NA" & CZ_NAME == "NA" ~ "DISTRICT OF COLUMBIA",
    
    # --- Global Fixes (Safe for all states) ---
    CZ_NAME == "DE KALB"  ~ "DEKALB",
    CZ_NAME == "DU PAGE"  ~ "DUPAGE",
    CZ_NAME == "LA PORTE" ~ "LAPORTE",
    CZ_NAME == "LA MOURE" ~ "LAMOURE",
    CZ_NAME == "DONA ANA" ~ "DOÑA ANA",
    
    # Virginia Independent Cities
    STATE == "VIRGINIA" & CZ_NAME == "CHESAPEAKE (C)"     ~ "CHESAPEAKE",
    STATE == "VIRGINIA" & CZ_NAME == "HAMPTON (C)"        ~ "HAMPTON",
    STATE == "VIRGINIA" & CZ_NAME == "NORFOLK (C)"        ~ "NORFOLK",
    STATE == "VIRGINIA" & CZ_NAME == "RICHMOND (C)"       ~ "RICHMOND",
    STATE == "VIRGINIA" & CZ_NAME == "VIRGINIA BEACH (C)" ~ "VIRGINIA BEACH",
    STATE == "VIRGINIA" & CZ_NAME == "NEWPORT NEWS (C)"   ~ "NEWPORT NEWS",
    STATE == "VIRGINIA" & CZ_NAME == "JAMES CITY" ~ "JAMES",
    STATE == "VIRGINIA" & CZ_NAME == "SUFFOLK (C)"  ~ "SUFFOLK",
    
    # Baltimore & St. Louis
    STATE == "MARYLAND" & CZ_NAME %in% c("BALTIMORE CITY (C)", "BALTIMORE CITY") ~ "BALTIMORE",
    STATE == "MISSOURI" & CZ_NAME %in% c("ST. LOUIS (C)", "ST. LOUIS CITY") ~ "ST. LOUIS",
    
    # Menominee Wisconsin
    STATE == "WISCONSIN" & CZ_NAME == "MENOMINEE (C)" ~ "MENOMINEE",
    
    # Desoto florida
    STATE == "FLORIDA"  & CZ_NAME == "DE SOTO" ~ "DESOTO",
    
    # Default: Keep original if no rules match
    TRUE ~ CZ_NAME),
    
    # Adjust state to NA for D.C.
    STATE = if_else(STATE == "DISTRICT OF COLUMBIA", "NA", STATE)
  ) |>
  mutate(county_id = paste(CZ_NAME, STATE, sep = "_"))

# Redo join with fixed names
gev_all <- gev_all |>
  select(-any_of(names(nri_clean)[names(nri_clean) != "county_id"])) |>
  left_join(nri_clean, by = "county_id")

cat("Counties with NRI exposure data:", sum(!is.na(gev_all$bldg_exposure)), "\n")
cat("Counties missing NRI exposure:", sum(is.na(gev_all$bldg_exposure)), "\n")

# Remove duplicate columns from previous join attempt
gev_all <- gev_all |>
  select(-any_of(c("prop_damage_ratio", "ag_damage_ratio",
                   "prop_damage_mean", "ag_damage_mean")))

# Now apply damage ratios cleanly
gev_all <- gev_all |>
  left_join(damage_ratios |> select(bin, prop_damage_mean, ag_damage_mean),
            by = c("bin_100" = "bin")) |>
  rename(
    prop_damage_ratio = prop_damage_mean,
    ag_damage_ratio   = ag_damage_mean
  )

# Quick check
cat("Counties with damage ratios assigned:", sum(!is.na(gev_all$prop_damage_ratio)), "\n")
summary(gev_all$prop_damage_ratio)

# Part 9: Loss Estimation

# Compute Bin Multipliers from NOAA Data
overall_mean_prop = mean(hail_data_1Plus$DAMAGE_PROPERTY, na.rm = TRUE)
overall_mean_ag   = mean(hail_data_1Plus$DAMAGE_CROPS,    na.rm = TRUE)

bin_multipliers <- damage_ratios |>
  mutate(
    multiplier_prop = prop_damage_mean / overall_mean_prop,
    multiplier_ag   = ag_damage_mean   / overall_mean_ag
  )

cat("Bin Multipliers:\n")
print(bin_multipliers |> select(bin, multiplier_prop, multiplier_ag))

# Join Bin Multipliers to gev_all
gev_all <- gev_all |>
  select(-any_of(c("prop_damage_ratio", "ag_damage_ratio",
                   "prop_damage_mean", "ag_damage_mean",
                   "loss_bldg_100", "loss_ag_100", "loss_total_100",
                   "loss_bldg_50", "loss_ag_50", "loss_total_50",
                   "loss_bldg_500", "loss_ag_500", "loss_total_500",
                   "fema_eal_total", "loss_vs_fema", "fema_comparison",
                   "multiplier_prop", "multiplier_ag"))) |>
  left_join(
    bin_multipliers |> select(bin, prop_damage_mean, ag_damage_mean,
                              multiplier_prop, multiplier_ag),
    by = c("bin_100" = "bin")
  )

# Compute County Level Losses
# Loss ($) = Exposure ($) x FEMA Loss Ratio x Bin Multiplier
gev_all <- gev_all |>
  mutate(
    # 100-year losses
    loss_bldg_100  = bldg_exposure * loss_ratio_bldg * multiplier_prop,
    loss_ag_100    = ag_exposure * loss_ratio_ag * multiplier_ag,
    loss_total_100 = loss_bldg_100 + loss_ag_100,
    # 50-year losses
    loss_bldg_50   = bldg_exposure * loss_ratio_bldg * multiplier_prop * (log(50)  / log(100)),
    loss_ag_50     = ag_exposure  * loss_ratio_ag * multiplier_ag   * (log(50)  / log(100)),
    loss_total_50  = loss_bldg_50  + loss_ag_50,
    # 500-year losses
    loss_bldg_500  = bldg_exposure * loss_ratio_bldg * multiplier_prop * (log(500) / log(100)),
    loss_ag_500    = ag_exposure   * loss_ratio_ag * multiplier_ag   * (log(500) / log(100)),
    loss_total_500 = loss_bldg_500 + loss_ag_500
  )

# Quick check
cat("Counties with loss estimates:", sum(!is.na(gev_all$loss_total_100)), "\n")
cat("Loss range ($M):",
    min(gev_all$loss_total_100, na.rm = TRUE) / 1e6, "to",
    max(gev_all$loss_total_100, na.rm = TRUE) / 1e6, "\n")
cat("Total modeled loss ($B):",
    sum(gev_all$loss_total_100, na.rm = TRUE) / 1e9, "\n")

# Summary of Losses by Model Type
gev_all |>
  filter(!is.na(loss_total_100)) |>
  group_by(model) |>
  summarise(
    n_counties      = n(),
    mean_loss_100   = mean(loss_total_100,   na.rm = TRUE),
    median_loss_100 = median(loss_total_100, na.rm = TRUE),
    total_loss_100  = sum(loss_total_100,    na.rm = TRUE),
    max_loss_100    = max(loss_total_100,    na.rm = TRUE)
  ) |>
  mutate(across(where(is.numeric), ~fmt_dollar(.)))

# Comparison Against FEMA EAL Benchmark
gev_all <- gev_all |>
  mutate(
    fema_eal_total  = fema_eal_bldg + fema_eal_ag,
    loss_vs_fema    = loss_total_100 / fema_eal_total,
    fema_comparison = case_when(
      loss_vs_fema > 1 ~ "Higher than FEMA",
      loss_vs_fema < 1 ~ "Lower than FEMA",
      TRUE             ~ "Equal to FEMA"
    )
  )

cat("\nFEMA Comparison:\n")
gev_all |>
  filter(!is.na(loss_vs_fema)) |>
  count(fema_comparison)

cat("\nMean ratio of modeled loss to FEMA EAL:\n")
summary(gev_all$loss_vs_fema)

# Loss Exceedance Curve
return_periods_plot <- c(2, 5, 10, 25, 50, 100, 200, 500, 1000)

loss_exceedance <- map_dfr(return_periods_plot, function(T) {
  
  gev_all |>
    filter(!is.na(bldg_exposure)) |>
    mutate(
      rl_T = case_when(
        T <= 100 ~ rl_50  + (rl_100 - rl_50)  * (log(T) - log(50))  / (log(100) - log(50)),
        T <= 500 ~ rl_100 + (rl_500 - rl_100) * (log(T) - log(100)) / (log(500) - log(100)),
        TRUE     ~ rl_500 + (rl_500 - rl_100) * (log(T) - log(500)) / (log(500) - log(100))
      ),
      bin_T = case_when(
        rl_T >= 1.0 & rl_T < 2.0 ~ "1.0-2.0 in",
        rl_T >= 2.0 & rl_T < 3.0 ~ "2.0-3.0 in",
        rl_T >= 3.0 & rl_T < 4.0 ~ "3.0-4.0 in",
        rl_T >= 4.0               ~ "4.0+ in",
        TRUE                      ~ NA_character_
      )
    ) |>
    left_join(
      bin_multipliers |>
        select(bin,
               multiplier_prop_T = multiplier_prop,
               multiplier_ag_T   = multiplier_ag),
      by = c("bin_T" = "bin")
    ) |>
    mutate(
      loss_bldg_T  = bldg_exposure * loss_ratio_bldg * multiplier_prop_T,
      loss_ag_T    = ag_exposure * loss_ratio_ag * multiplier_ag_T,
      loss_total_T = loss_bldg_T  + loss_ag_T
    ) |>
    summarise(
      return_period = T,
      total_loss    = sum(loss_total_T, na.rm = TRUE) / 1e9,
      exceed_prob   = 1 / T
    )
})

cat("\nLoss Exceedance Summary ($B):\n")
print(loss_exceedance)


# Plot
plot(loss_exceedance$return_period, loss_exceedance$total_loss,
     log  = "x",
     type = "b",
     lwd  = 2,
     pch  = 19,
     col  = "steelblue",
     xlab = "Return Period (years)",
     ylab = "Total U.S. Hail Loss ($B)",
     main = "Total Loss Exceedance Curve - U.S. Hail Risk",
     ylim = c(0, max(loss_exceedance$total_loss) * 1.1))
grid()

lines(loss_exceedance$return_period, loss_exceedance$total_loss,
      col = "steelblue", lwd = 2)

# FEMA EAL reference
abline(h = sum(gev_all$fema_eal_total, na.rm = TRUE) / 1e9,
       col = "red", lwd = 2, lty = 2)

abline(v = 50,  col = "orange", lwd = 1.5, lty = 3)
abline(v = 100, col = "red",    lwd = 1.5, lty = 3)
abline(v = 500, col = "purple", lwd = 1.5, lty = 3)

legend("topleft",
       legend = c("Modeled Loss", "FEMA Total EAL", "50-yr", "100-yr", "500-yr"),
       col    = c("steelblue", "red", "orange", "red", "purple"),
       pch    = c(19, NA, NA, NA, NA),
       lwd    = c(2, 2, 1.5, 1.5, 1.5),
       lty    = c(1, 2, 3, 3, 3))

# Top 10 Highest Loss 

# Ensure no duplicate ID's appear in the rankings
gev_all <- gev_all |>
  distinct(county_id, .keep_all = TRUE)

# Top 10 by Building Loss
top_building_loss <- gev_all |>
  filter(!is.na(loss_bldg_100)) |>
  distinct(county_id, .keep_all = TRUE) |>
  arrange(desc(loss_bldg_100)) |>
  mutate(
    Rank       = row_number(),
    County     = paste(CZ_NAME, STATE, sep = ", "),
    Model      = model,
    Hail_Bin   = bin_100,
    Bldg_Loss  = fmt_dollar(loss_bldg_100),
    Ag_Loss    = fmt_dollar(loss_ag_100),
    Total_Loss = fmt_dollar(loss_total_100),
    FEMA_EAL   = fmt_dollar(fema_eal_total),
    vs_FEMA    = paste0(round(loss_vs_fema, 1), "x")
  ) |>
  select(Rank, County, Model, Hail_Bin,
         Bldg_Loss, Ag_Loss, Total_Loss,
         FEMA_EAL, vs_FEMA) |>
  head(10)

print(top_building_loss)

# Top 10 by Agricultural Loss
top_agricultural_loss <- gev_all |>
  filter(!is.na(loss_ag_100)) |>
  distinct(county_id, .keep_all = TRUE) |>
  arrange(desc(loss_ag_100)) |>
  mutate(
    Rank       = row_number(),
    County     = paste(CZ_NAME, STATE, sep = ", "),
    Model      = model,
    Hail_Bin   = bin_100,
    Bldg_Loss  = fmt_dollar(loss_bldg_100),
    Ag_Loss    = fmt_dollar(loss_ag_100),
    Total_Loss = fmt_dollar(loss_total_100),
    FEMA_EAL   = fmt_dollar(fema_eal_total),
    vs_FEMA    = paste0(round(loss_vs_fema, 1), "x")
  ) |>
  select(Rank, County, Model, Hail_Bin,
         Bldg_Loss, Ag_Loss, Total_Loss,
         FEMA_EAL, vs_FEMA) |>
  head(10)

print(top_agricultural_loss)

# Top 10 by Total Loss
top_total_loss <- gev_all |>
  filter(!is.na(loss_total_100)) |>
  distinct(county_id, .keep_all = TRUE) |>
  arrange(desc(loss_total_100)) |>
  mutate(
    Rank       = row_number(),
    County     = paste(CZ_NAME, STATE, sep = ", "),
    Model      = model,
    Hail_Bin   = bin_100,
    Bldg_Loss  = fmt_dollar(loss_bldg_100),
    Ag_Loss    = fmt_dollar(loss_ag_100),
    Total_Loss = fmt_dollar(loss_total_100),
    FEMA_EAL   = fmt_dollar(fema_eal_total),
    vs_FEMA    = paste0(round(loss_vs_fema, 1), "x")
  ) |>
  select(Rank, County, Model, Hail_Bin,
         Bldg_Loss, Ag_Loss, Total_Loss,
         FEMA_EAL, vs_FEMA) |>
  head(10)

print(top_total_loss)

# State-Level Total Loss Summary
state_losses <- gev_all |>
  filter(!is.na(loss_total_100)) |>
  distinct(county_id, .keep_all = TRUE) |>
  group_by(STATE) |>
  summarise(
    n_counties     = n(),
    total_bldg     = sum(loss_bldg_100,  na.rm = TRUE),
    total_ag       = sum(loss_ag_100,    na.rm = TRUE),
    total_loss     = sum(loss_total_100, na.rm = TRUE),
    fema_total     = sum(fema_eal_total, na.rm = TRUE)
  ) |>
  arrange(desc(total_loss)) |>
  mutate(
    Rank       = row_number(),
    State      = STATE,
    N_Counties = n_counties,
    Bldg_Loss  = fmt_dollar(total_bldg),
    Ag_Loss    = fmt_dollar(total_ag),
    Total_Loss = fmt_dollar(total_loss),
    FEMA_EAL   = fmt_dollar(fema_total),
    vs_FEMA    = paste0(round(total_loss / fema_total, 1), "x")
  ) |>
  select(Rank, State, N_Counties, Bldg_Loss, 
         Ag_Loss, Total_Loss, FEMA_EAL, vs_FEMA) |>
  head(10)

print(state_losses)

# Top 10 by largest modeled loss vs FEMA EAL
top_ten_fema <- gev_all |>
  filter(!is.na(loss_vs_fema)) |>
  arrange(desc(loss_vs_fema)) |>
  mutate(
    County     = paste(CZ_NAME, STATE, sep = ", "),
    Model = model,
    Total_Loss = fmt_dollar(loss_total_100),
    FEMA_EAL   = fmt_dollar(fema_eal_total),
    vs_FEMA    = paste0(round(loss_vs_fema, 1), "x")
  ) |>
  select(County, bin_100, Model, Total_Loss, FEMA_EAL, vs_FEMA) |>
  head(10)

print(top_ten_fema)

cat("Total FEMA EAL:", 
    fmt_dollar(sum(gev_all$fema_eal_total, na.rm = TRUE)), "\n")

# Part 10: Loss Summary Maps

clean_county_name_loss <- function(region, subregion) {
  x <- tolower(subregion)
  
  # State-specific fixes only
  x <- case_when(
    region == "florida"      & x == "de soto"        ~ "desoto",
    region == "illinois"     & x == "du page"        ~ "dupage",
    region == "illinois"     & x == "la salle"       ~ "lasalle",
    region == "indiana"      & x == "la porte"       ~ "laporte",
    region == "louisiana"    & x == "la salle"       ~ "lasalle",
    region == "maryland"     & x == "baltimore city" ~ "baltimore",
    region == "mississippi"  & x == "de soto"        ~ "desoto",
    region == "new mexico"   & x == "dona ana"       ~ "doña ana",
    region == "north dakota" & x == "la moure"       ~ "lamoure",
    region == "texas"        & x == "de witt"        ~ "dewitt",
    region == "virginia"     & x == "suffolk"        ~ "suffolk",
    region == "virginia"     & x == "james city"     ~ "james",
    TRUE ~ x
  )
  return(x)
}

# Prepare loss data for mapping
loss_map <- gev_all |>
  distinct(county_id, .keep_all = TRUE) |>
  filter(!is.na(loss_total_100)) |>
  mutate(
    state_lower  = tolower(STATE),
    county_lower = clean_county_name(CZ_NAME),
    # Convert CT planning regions back to old county names
    county_lower = case_when(
      state_lower == "connecticut" & county_lower == "southeastern connecticut"       ~ "new london",
      state_lower == "connecticut" & county_lower == "northeastern connecticut"       ~ "windham",
      state_lower == "connecticut" & county_lower == "capitol"                        ~ "hartford",
      state_lower == "connecticut" & county_lower == "northwest hills"                ~ "litchfield",
      state_lower == "connecticut" & county_lower == "lower connecticut river valley" ~ "middlesex",
      state_lower == "connecticut" & county_lower == "south central connecticut"      ~ "new haven",
      state_lower == "connecticut" & county_lower == "western connecticut"            ~ "fairfield",
      TRUE ~ county_lower
    )
  ) |>
  distinct(state_lower, county_lower, .keep_all = TRUE)

# Tolland shares Capitol with Hartford so add as duplicate
tolland_row <- loss_map |>
  filter(state_lower == "connecticut" & county_lower == "hartford") |>
  mutate(county_lower = "tolland")

# New Haven also covers Naugatuck Valley so add duplicate
newhaven_row <- loss_map |>
  filter(state_lower == "connecticut" & county_lower == "new haven") |>
  mutate(county_lower = "naugatuck valley")

# Fairfield also covers Greater Bridgeport so add duplicate  
fairfield_row <- loss_map |>
  filter(state_lower == "connecticut" & county_lower == "fairfield") |>
  mutate(county_lower = "greater bridgeport")

loss_map <- bind_rows(loss_map, tolland_row, newhaven_row, fairfield_row)

# Get fresh map data
us_states <- map_data("state")

# Building Loss (100-Year Return Period)

us_counties_bldg <- map_data("county") |>
  mutate(subregion_clean = clean_county_name_loss(region, subregion)) |>
  left_join(loss_map |> select(state_lower, county_lower, loss_bldg_100),
            by = c("region" = "state_lower", "subregion_clean" = "county_lower")) |>
  mutate(loss_bldg_100 = loss_bldg_100 / 1e6)  # convert to $M

n_bldg_shaded <- us_counties_bldg |>
  filter(!is.na(loss_bldg_100)) |>
  distinct(region, subregion) |>
  nrow()

ggplot() +
  geom_polygon(data = us_counties_bldg,
               aes(x = long, y = lat, group = group,
                   fill = loss_bldg_100),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_gradientn(
    colors   = c("cyan", "royalblue", "#081d58"),
    na.value = "grey90",
    name     = "Building\nLoss ($M)",
    trans    = "log10",
    labels   = scales::comma
  ) +
  coord_fixed(1.3) +
  labs(
    title    = "Estimated 100-Year Building Loss from Hail",
    subtitle = paste0("Total modeled building loss: ",
                      fmt_dollar(sum(gev_all$loss_bldg_100, na.rm = TRUE)),
                      " | n = ", n_bldg_shaded, " counties"),
    caption  = "Source: NOAA Storm Events Database & FEMA NRI"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "right"
  )

# Agricultural Loss (100-Year Return Period)

us_counties_ag <- map_data("county") |>
  mutate(subregion_clean = clean_county_name_loss(region, subregion)) |>
  left_join(loss_map |> select(state_lower, county_lower, loss_ag_100),
            by = c("region" = "state_lower", "subregion_clean" = "county_lower")) |>
  mutate(loss_ag_100 = loss_ag_100 / 1e6)  # convert to $M

n_ag_shaded <- us_counties_ag |>
  filter(!is.na(loss_ag_100)) |>
  distinct(region, subregion) |>
  nrow()

ggplot() +
  geom_polygon(data = us_counties_ag,
               aes(x = long, y = lat, group = group,
                   fill = loss_ag_100),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_gradientn(
    colors   = c("lightyellow", "lightgreen", "darkgreen"),
    na.value = "grey90",
    name     = "Agricultural\nLoss ($M)",
    trans    = "log10",
    labels   = scales::comma
  ) +
  coord_fixed(1.3) +
  labs(
    title    = "Estimated 100-Year Agricultural Loss from Hail",
    subtitle = paste0("Total modeled agricultural loss: ",
                      fmt_dollar(sum(gev_all$loss_ag_100, na.rm = TRUE)),
                      " | n = ", n_ag_shaded, " counties"),
    caption  = "Source: NOAA Storm Events Database & FEMA NRI"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "right"
  )

# Total Loss (100-Year Return Period)

us_counties_total <- map_data("county") |>
  mutate(subregion_clean = clean_county_name_loss(region, subregion)) |>
  left_join(loss_map |> select(state_lower, county_lower, loss_total_100),
            by = c("region" = "state_lower", "subregion_clean" = "county_lower")) |>
  mutate(loss_total_100 = loss_total_100 / 1e6)  # convert to $M

n_total_shaded <- us_counties_total |>
  filter(!is.na(loss_total_100)) |>
  distinct(region, subregion) |>
  nrow()

ggplot() +
  geom_polygon(data = us_counties_total,
               aes(x = long, y = lat, group = group,
                   fill = loss_total_100),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_gradientn(
    colors   = c("#fde0ef", "magenta", "#4a1486"),
    na.value = "grey90",
    name     = "Total\nLoss ($M)",
    trans    = "log10",
    labels   = scales::comma
  ) +
  coord_fixed(1.3) +
  labs(
    title    = "Estimated 100-Year Total Hail Loss (Buildings + Agriculture)",
    subtitle = paste0("Total modeled loss: ",
                      fmt_dollar(sum(gev_all$loss_total_100, na.rm = TRUE)),
                      " | n = ", n_total_shaded, " counties"),
    caption  = "Source: NOAA Storm Events Database & FEMA NRI"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "right"
  )

# Modeled Loss vs FEMA EAL

us_counties_fema <- map_data("county") |>
  mutate(subregion_clean = clean_county_name_loss(region, subregion)) |>
  left_join(loss_map |> select(state_lower, county_lower,
                               fema_comparison, loss_vs_fema),
            by = c("region" = "state_lower", "subregion_clean" = "county_lower")) |>
  mutate(fema_comparison = ifelse(is.na(fema_comparison),
                                  "No Data", fema_comparison))

# Count shaded counties by fema comparison
shaded_fema <- us_counties_fema |>
  filter(fema_comparison != "No Data") |>
  distinct(region, subregion, .keep_all = TRUE) |>
  count(fema_comparison)

n_higher = shaded_fema$n[shaded_fema$fema_comparison == "Higher than FEMA"]
n_lower  = shaded_fema$n[shaded_fema$fema_comparison == "Lower than FEMA"]

ggplot() +
  geom_polygon(data = us_counties_fema,
               aes(x = long, y = lat, group = group,
                   fill = fema_comparison),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_manual(
    values = c("Higher than FEMA" = "firebrick",
               "Lower than FEMA"  = "steelblue",
               "No Data"          = "grey90"),
    labels = c("Higher than FEMA" = "Higher than FEMA EAL",
               "Lower than FEMA"  = "Lower than FEMA EAL",
               "No Data"          = "No Data"),
    name = "vs FEMA EAL"
  ) +
  coord_fixed(1.3) +
  labs(
    title    = "Modeled 100-Year Loss vs FEMA Expected Annual Loss",
    subtitle = paste0("Higher than FEMA: ", n_higher,
                      " counties  |  Lower than FEMA: ", n_lower, " counties"),
    caption  = "Source: NOAA Storm Events Database & FEMA NRI"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "bottom"
  )

# State-Level Total Loss Map

# Compute total loss for lower 48 states - filter BEFORE ranking
state_loss_all <- gev_all |>
  filter(!is.na(loss_total_100)) |>
  distinct(county_id, .keep_all = TRUE) |>
  filter(!STATE %in% c("ALASKA", "HAWAII", "NA")) |>
  group_by(STATE) |>
  summarise(
    total_loss = sum(loss_total_100, na.rm = TRUE)
  ) |>
  arrange(desc(total_loss)) |>
  mutate(
    rank        = row_number(),
    state_lower = tolower(STATE)
  )

cat("States ranked:", nrow(state_loss_all), "\n")
cat("Max rank:", max(state_loss_all$rank), "\n")

# Join with state map data
us_states_loss <- map_data("state") |>
  left_join(state_loss_all,
            by = c("region" = "state_lower")) |>
  mutate(rank = ifelse(is.na(rank), NA, rank))

# Hardcoded geographic centroids
state_centroids <- data.frame(
  region = c("alabama", "arizona", "arkansas", "california", "colorado",
             "connecticut", "delaware", "florida", "georgia", "idaho",
             "illinois", "indiana", "iowa", "kansas", "kentucky",
             "louisiana", "maine", "maryland", "massachusetts", "michigan",
             "minnesota", "mississippi", "missouri", "montana", "nebraska",
             "nevada", "new hampshire", "new jersey", "new mexico", "new york",
             "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
             "pennsylvania", "rhode island", "south carolina", "south dakota",
             "tennessee", "texas", "utah", "vermont", "virginia",
             "washington", "west virginia", "wisconsin", "wyoming"),
  long   = c(-86.8, -111.7, -92.4, -119.5, -105.5,
             -72.7, -75.5, -81.5, -83.4, -114.5,
             -89.2, -86.3, -93.5, -98.4, -84.3,
             -91.8, -69.2, -76.8, -71.8, -84.5,
             -94.3, -89.7, -92.5, -110.5, -99.9,
             -116.8, -71.6, -74.5, -106.1, -75.5,
             -79.4, -100.5, -82.8, -97.5, -120.5,
             -77.2, -71.5, -80.9, -100.3, -86.3,
             -99.3, -111.5, -72.7, -78.5, -120.5,
             -80.5, -89.7, -107.5),
  lat    = c(32.8, 34.3, 34.9, 37.2, 39.0,
             41.6, 39.0, 28.1, 32.7, 44.4,
             40.0, 40.3, 42.1, 38.5, 37.5,
             31.2, 45.4, 39.0, 42.3, 44.3,
             46.4, 32.7, 38.5, 47.0, 41.5,
             39.5, 44.0, 40.1, 34.4, 43.0,
             35.5, 47.5, 40.4, 35.6, 44.1,
             40.9, 41.7, 33.9, 44.4, 35.9,
             31.5, 39.4, 44.0, 37.5, 47.4,
             38.9, 44.5, 43.1)
) |>
  left_join(state_loss_all |> select(state_lower, rank),
            by = c("region" = "state_lower")) |>
  filter(!is.na(rank))

# Plot
ggplot() +
  geom_polygon(data = us_states_loss,
               aes(x = long, y = lat, group = group,
                   fill = rank),
               color = "white", linewidth = 0.3) +
  scale_fill_gradientn(
    colors   = c("darkred", "red", "orange", "lightyellow"),
    na.value = "grey90",
    name     = "Rank\n(1 = Highest)",
    breaks   = c(1, 10, 20, 30, 40, 48),
    labels   = c("1", "10", "20", "30", "40", "48")
  ) +
  geom_text(data = state_centroids,
            aes(x = long, y = lat, label = rank),
            size     = 4,
            fontface = "bold",
            color    = "black") +
  coord_fixed(1.3) +
  labs(
    title    = "U.S. State Hail Risk Ranking by 100-Year Total Loss (Lower 48)",
    subtitle = paste0("Ranked by estimated building + agricultural loss at 100-year return period\n",
                      "Total modeled U.S. loss: ",
                      fmt_dollar(sum(gev_all$loss_total_100, na.rm = TRUE))),
    caption  = paste0("Source: NOAA Storm Events Database & FEMA NRI")
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "right"
  )

# 100 year hail size bin map

# Prepare bin data for mapping
bin_map <- bind_rows(
  gev_stationary |>
    mutate(
      state_lower  = tolower(STATE),
      county_lower = clean_county_name(CZ_NAME)
    ) |>
    select(state_lower, county_lower, bin_100),
  gev_nonstationary |>
    mutate(
      state_lower  = tolower(STATE),
      county_lower = clean_county_name(CZ_NAME)
    ) |>
    select(state_lower, county_lower, bin_100)
) |>
  distinct(state_lower, county_lower, .keep_all = TRUE)

# Get fresh county map data
us_counties_bin <- map_data("county") |>
  left_join(bin_map,
            by = c("region"    = "state_lower",
                   "subregion" = "county_lower")) |>
  mutate(bin_100 = ifelse(is.na(bin_100), "No Data", bin_100))

# Count shaded counties by bin
shaded_bin <- us_counties_bin |>
  filter(bin_100 != "No Data") |>
  distinct(region, subregion, .keep_all = TRUE) |>
  count(bin_100)

print(shaded_bin)

# Plot
ggplot() +
  geom_polygon(data = us_counties_bin,
               aes(x = long, y = lat, group = group,
                   fill = bin_100),
               color = "white", linewidth = 0.1) +
  geom_polygon(data = us_states,
               aes(x = long, y = lat, group = group),
               fill = NA, color = "black", linewidth = 0.3) +
  scale_fill_manual(
    values = c("1.0-2.0 in" = "lightyellow",
               "2.0-3.0 in" = "orange",
               "3.0-4.0 in" = "red",
               "4.0+ in"    = "darkred",
               "No Data"    = "grey90"),
    labels = c("1.0-2.0 in" = "1.0-2.0 inches",
               "2.0-3.0 in" = "2.0-3.0 inches",
               "3.0-4.0 in" = "3.0-4.0 inches",
               "4.0+ in"    = "4.0+ inches",
               "No Data"    = "No Data"),
    name = "100-Year\nHail Size"
  ) +
  coord_fixed(1.3) +
  labs(
    title    = "100-Year Hail Size Category Across U.S. Counties",
    subtitle = paste0("Hail diameter bin classification based on GEV return level estimates | n = ",
                      sum(shaded_bin$n)),
    caption  = "Source: NOAA Storm Events Database (1950-2024)"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "right"
  )