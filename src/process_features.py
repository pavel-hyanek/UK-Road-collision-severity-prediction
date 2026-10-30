import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


collision_df = pd.read_csv(PROJECT_ROOT / 'data/raw_data/collision_raw.csv', low_memory=False)
vehicle_df = pd.read_csv(PROJECT_ROOT / 'data/raw_data/vehicle_raw.csv', low_memory=False)

collision_df["date_month"] = pd.to_datetime(collision_df["date"], dayfirst=True).dt.month
collision_df["time_hour"] = pd.to_datetime(collision_df["time"], format="%H:%M").dt.hour
change_of_collision_severity = {1 : 1, 2 : 1, 3 : 0}
collision_df["severity"] = collision_df["collision_severity"].map(change_of_collision_severity)
collision_df["number_of_vehicles"] = collision_df["number_of_vehicles"].astype(np.int16)
collision_df["is_weekend"] = collision_df["day_of_week"].map((lambda x: 1 if x in [1,7] else 0))

unite_missing_category_collision_columns = [("speed_limit", 99), ("junction_detail_historic", 99),
                                            ("road_type", 9), ("speed_limit", 99),
                                            ("junction_detail", 99), ("junction_control", 9),
                                            ("second_road_class", 9), ("pedestrian_crossing", 99),
                                            ("weather_conditions", 9), ("road_surface_conditions", 9),
                                            ("carriageway_hazards_historic", 9), ("special_conditions_at_site", 9)]
unite_missing_category_vehicle_columns = [("vehicle_manoeuvre", 99), ("vehicle_location_restricted_lane_historic", 99),
                                         ("junction_location", 9), ("skidding_and_overturning", 9),
                                         ("hit_object_in_carriageway", 99), ("vehicle_leaving_carriageway", 9),
                                         ("hit_object_off_carriageway", 99), ("first_point_of_impact", 9)]

for column, n in unite_missing_category_collision_columns:
    collision_df[column] = collision_df[column].map((lambda x: -1 if x==n else x))

for column, n in unite_missing_category_vehicle_columns:
    vehicle_df[column] = vehicle_df[column].map((lambda x: -1 if x==n else x))

for column in vehicle_df.columns:
    if column not in ["collision_year", "collision_index"]:
        vehicle_df.rename(columns={column : "VEHICLE_" + column}, inplace=True)

collision_df.rename(columns={"number_of_vehicles" : "VEHICLE_number_of_vehicles"}, inplace=True)


collision_columns_impute_and_encode = ["police_force", "date_month", "time_hour", "first_road_class", "road_type", 
                     "speed_limit", "junction_detail", "second_road_class", "light_conditions", 
                       "weather_conditions", "road_surface_conditions", "urban_or_rural_area"]

#collision_columns_impute_missing = ["junction_detail", "second_road_class", "road_surface_conditions"]

collision_columns = ["collision_index", "severity", "VEHICLE_number_of_vehicles"]

vehicle_columns_impute_and_encode = ["VEHICLE_vehicle_type", "VEHICLE_skidding_and_overturning", "VEHICLE_hit_object_in_carriageway",
                   "VEHICLE_hit_object_off_carriageway", "VEHICLE_vehicle_leaving_carriageway", 
                   "VEHICLE_vehicle_manoeuvre", "VEHICLE_first_point_of_impact"]

#vehicle_columns_impute_missing = ["VEHICLE_skidding_and_overturning", "VEHICLE_hit_object_in_carriageway",
#                                  "VEHICLE_hit_object_off_carriageway", "VEHICLE_vehicle_leaving_carriageway",
#                                  "VEHICLE_vehicle_manoeuvre", "VEHICLE_first_point_of_impact"]

vehicle_columns = ["collision_index"]

collision_train = collision_df[collision_df["collision_year"]<=2023][collision_columns_impute_and_encode+collision_columns]
collision_train[collision_columns_impute_and_encode] = collision_train[collision_columns_impute_and_encode].fillna(-1).astype(str)
collision_valid = collision_df[collision_df["collision_year"]==2024][collision_columns_impute_and_encode+collision_columns]
collision_valid[collision_columns_impute_and_encode] = collision_valid[collision_columns_impute_and_encode].fillna(-1).astype(str)
collision_test = collision_df[collision_df["collision_year"]==2025][collision_columns_impute_and_encode+collision_columns]
collision_test[collision_columns_impute_and_encode] = collision_test[collision_columns_impute_and_encode].fillna(-1).astype(str)

vehicle_train = vehicle_df[vehicle_df["collision_year"]<=2023][vehicle_columns_impute_and_encode+vehicle_columns]
vehicle_train[vehicle_columns_impute_and_encode] = vehicle_train[vehicle_columns_impute_and_encode].fillna(-1).astype(str)
vehicle_valid = vehicle_df[vehicle_df["collision_year"]==2024][vehicle_columns_impute_and_encode+vehicle_columns]
vehicle_valid[vehicle_columns_impute_and_encode] = vehicle_valid[vehicle_columns_impute_and_encode].fillna(-1).astype(str)
vehicle_test = vehicle_df[vehicle_df["collision_year"]==2025][vehicle_columns_impute_and_encode+vehicle_columns]
vehicle_test[vehicle_columns_impute_and_encode] = vehicle_test[vehicle_columns_impute_and_encode].fillna(-1).astype(str)

collision_train_final = pd.concat([collision_train, collision_valid])
vehicle_train_final = pd.concat([vehicle_train, vehicle_valid])

collision_imputer = SimpleImputer(missing_values="-1", strategy="most_frequent")
vehicle_imputer = SimpleImputer(missing_values="-1", strategy="most_frequent")

collision_imputer.fit(collision_train[collision_columns_impute_and_encode])
vehicle_imputer.fit(vehicle_train[vehicle_columns_impute_and_encode])

collision_imputer_final = SimpleImputer(missing_values="-1", strategy="most_frequent")
vehicle_imputer_final = SimpleImputer(missing_values="-1", strategy="most_frequent")

collision_imputer_final.fit(collision_train_final[collision_columns_impute_and_encode])
vehicle_imputer_final.fit(vehicle_train_final[vehicle_columns_impute_and_encode])


collision_encoder = OneHotEncoder(dtype=np.int16, handle_unknown="ignore", sparse_output=False, 
                                  feature_name_combiner="concat")
vehicle_encoder = OneHotEncoder(dtype=np.int16, handle_unknown="ignore", sparse_output=False, 
                                feature_name_combiner="concat")

collision_encoder.fit(collision_train[collision_columns_impute_and_encode])
vehicle_encoder.fit(vehicle_train[vehicle_columns_impute_and_encode])

collision_encoder_final = OneHotEncoder(dtype=np.int16, handle_unknown="ignore", sparse_output=False, 
                                        feature_name_combiner="concat")
vehicle_encoder_final = OneHotEncoder(dtype=np.int16, handle_unknown="ignore", sparse_output=False, 
                                      feature_name_combiner="concat")

collision_encoder_final.fit(collision_train_final[collision_columns_impute_and_encode])
vehicle_encoder_final.fit(vehicle_train_final[vehicle_columns_impute_and_encode])

# to prevent leakage the imputers and encoders are fit only on the train instances, 
# although the leakage would be almost negligible if it happened

def impute_encode_collision(df, impute_columns=collision_columns_impute_and_encode, 
                            encode_columns=collision_columns_impute_and_encode, imputer=collision_imputer, 
                            encoder=collision_encoder):
    df = df.copy()
    df[impute_columns] = imputer.transform(df[impute_columns])  
    df_encoded = pd.DataFrame(encoder.transform(df[encode_columns]))
    df_encoded.index = df.index
    df_encoded.columns=encoder.get_feature_names_out(encode_columns)
    return pd.concat([df_encoded,df.drop(encode_columns, axis=1)], axis=1)

def impute_encode_vehicle(df, impute_columns=vehicle_columns_impute_and_encode, 
                          encode_columns=vehicle_columns_impute_and_encode, imputer=vehicle_imputer,
                          encoder=vehicle_encoder):
    df = df.copy()
    df[impute_columns] = imputer.transform(df[impute_columns])
    df_encoded = pd.DataFrame(encoder.transform(df[encode_columns]))
    df_encoded.index = df.index
    df_encoded.columns=encoder.get_feature_names_out(encode_columns)
    ret = pd.concat([df_encoded,df.drop(encode_columns, axis=1).astype(str)], axis=1)
    ret = ret.groupby(["collision_index"]).sum().reset_index()
    return ret


collision_train_encoded = impute_encode_collision(collision_train)
collision_valid_encoded = impute_encode_collision(collision_valid)
collision_test_encoded = impute_encode_collision(collision_test)

vehicle_train_encoded = impute_encode_vehicle(vehicle_train)
vehicle_valid_encoded = impute_encode_vehicle(vehicle_valid)
vehicle_test_encoded = impute_encode_vehicle(vehicle_test)

collision_train_final_encoded = impute_encode_collision(collision_train_final, imputer=collision_imputer_final, 
                                                        encoder=collision_encoder_final)
vehicle_train_final_encoded = impute_encode_vehicle(vehicle_train_final, imputer=vehicle_imputer_final,
                                                    encoder=vehicle_encoder_final)
collision_test_final_encoded = impute_encode_collision(collision_test, imputer=collision_imputer_final, 
                                                        encoder=collision_encoder_final)
vehicle_test_final_encoded = impute_encode_vehicle(vehicle_test, imputer=vehicle_imputer_final,
                                                    encoder=vehicle_encoder_final)

train = vehicle_train_encoded.merge(collision_train_encoded, on="collision_index", how="inner")
valid = vehicle_valid_encoded.merge(collision_valid_encoded, on="collision_index", how="inner")
test = vehicle_test_encoded.merge(collision_test_encoded, on="collision_index", how="inner")

train_final = vehicle_train_final_encoded.merge(collision_train_final_encoded, on="collision_index", how="inner")
test_final = vehicle_test_final_encoded.merge(collision_test_final_encoded, on="collision_index", how="inner")

train.to_parquet(PROJECT_ROOT / "data/preprocessed/train.parquet")
valid.to_parquet(PROJECT_ROOT / "data/preprocessed/valid.parquet")
test.to_parquet(PROJECT_ROOT / "data/preprocessed/test.parquet")

train_final.to_parquet(PROJECT_ROOT / "data/preprocessed/train_final.parquet")
test_final.to_parquet(PROJECT_ROOT / "data/preprocessed/test_final.parquet")