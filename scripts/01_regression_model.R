# 01_regression_model.R
# 06 - 29 - 21


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(textrecipes)
library(glmnet)
library(stopwords)
library(vip)
tidymodels_prefer()

# Data Import -------------------------------------------------------------

reviews <- read_csv(here::here("data/raw_data/reviews.csv"))
listings <- read_csv(here::here("data/raw_data/listings.csv"))

skimr::skim(reviews)
skimr::skim(listings)

# glad to see price isn't missing, but some reviews are clearly duplicates

# Data Exploration --------------------------------------------------------

# goal is to build a regression text model on review vs. price
model_df <- listings |>
  mutate(price = parse_number(price)) |>
  left_join(reviews, by = c("id" = "listing_id")) |>
  select(id, comments, price) |>
  filter(!is.na(comments))

# histogram of response variable, price
model_df |>
  ggplot(aes(x = price)) +
  geom_histogram()
# price is clearly not normal, need to consider transformations
model_df |>
  ggplot(aes(x = log(price, base = 2)))+
  geom_histogram()

model_df |>
  ggplot(aes(x = scale(price))) +
  geom_histogram()

model_df |>
  ggplot(aes(x = scale(log(price,base = 2)))) +
  geom_histogram()

# log 2 then scaling seems to produce the best results


# Model Setup -------------------------------------------------------------

set.seed(317)
# split keeping the listing id in mind
split <- initial_split(model_df, prop = 0.8, strata = id)
price_test <- testing(split)
price_train <- training(split)

set.seed(117)
price_folds <- vfold_cv(price_train, strata = id)
price_folds

price_rec <- recipe(price ~ comments, data = price_train) |>
  step_log(price, base = 2) |>
  step_scale(price) |>
  step_tokenize(comments) |>
  step_stopwords(comments) |>
  step_tokenfilter(comments, max_tokens = 500) |>
  step_tfidf(comments)


# Workflow ----------------------------------------------------------------

sparse_bp <- hardhat::default_recipe_blueprint(composition = "dgCMatrix")

lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

price_wf <- workflow() %>%
  add_recipe(price_rec, blueprint = sparse_bp) %>%
  add_model(lasso_spec)

price_wf

# Parameter Tuning --------------------------------------------------------

lambda_grid <- grid_regular(penalty(range = c(-3, 0)), levels = 20)

doParallel::registerDoParallel()
set.seed(517)

lasso_rs <- tune_grid(
  price_wf,
  resamples = price_folds,
  grid = lambda_grid
)

lasso_rs
autoplot(lasso_rs)

show_best(lasso_rs, "rmse")

# Select Model ------------------------------------------------------------

best_rmse <- select_best(lasso_rs, "rmse")

final_lasso <- finalize_workflow(price_wf, best_rmse)
final_lasso

price_final <- last_fit(final_lasso, split)
collect_metrics(price_final)

# Variable Importance Plots -----------------------------------------------

price_vip <- pull_workflow_fit(price_final$.workflow[[1]]) %>%
  vip::vi()

# code straight from Julia Silge
price_vip %>%
  group_by(Sign) %>%
  slice_max(abs(Importance), n = 20) %>%
  ungroup() %>%
  mutate(
    Variable = str_remove(Variable, "tfidf_comments_"),
    Importance = abs(Importance),
    Variable = fct_reorder(Variable, Importance),
    Sign = if_else(Sign == "POS", "Higher Price", "Lower Price")
  ) %>%
  ggplot(aes(Importance, Variable, fill = Sign)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Sign, scales = "free") +
  labs(y = NULL)

# check if this is right in terms of labelling


# Check Predictions -------------------------------------------------------

collect_predictions(price_final) %>%
  ggplot(aes(price, .pred)) +
  geom_abline(lty = 2, color = "gray50", size = 1.2) +
  geom_point(size = 1.5, alpha = 0.3, color = "midnightblue") +
  coord_fixed()
