# 01_regression_model.R
# 06 - 29 - 21


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
tidymodels_prefer()

# Data Import -------------------------------------------------------------

reviews <- read_csv(here::here("data/raw_data/reviews.csv"))
