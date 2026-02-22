# Simple vs. Optimal Congestion Pricing

This repository contains a Python 3 implementation for the paper titled "Simple vs. Optimal Congestion Pricing".

## Description
bay_bridge_data.ipynb: This file computes the average BART fare and average BART travel time for east to west bay trips using BART GTFS data, which requires the files "fare_rules.txt", "fare_attributes.txt", "routes.txt", "stop_times.txt", "stops.txt", and "trips.txt". Moreover, this file computes the average weekday demand in August 2025 using PEMS data and BART data, which requires input files "pems_output.xlsx" and "date-hour-soo-dest-2025.csv".

data_bay_bridge.zip: This folder contains the datasets used in bay_bridge_data.ipynb.

bay_bridge_bottleneck_comparisons.ipynb: This file compares the performance of static and dynamic tolling policies for the bay bridge case study under the bottleneck model with an outside option.

nyc_data.ipynb: This file calculates the total number of taxi trips using TLC data and the number of vehicle entries in the CRZ using vehicle entries data using a file "MTA_Congestion_Relief_Zone_Vehicle_Entries__Beginning_2025_20251106.csv".

data_mfd.zip: This folder contains the data used in nyc_data.ipynb.

subway_od_data.zip: This folder has the details on the calibration of the total subway trips within, to, and from the CRZ. The code and analysis for this was done by Mr. Luyang Han.

nyc_mfd_comparisons.ipynb: This file compares the performance of static and dynamic tolling policies for the nyc case study under the mfd model.
