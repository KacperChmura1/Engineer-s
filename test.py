
from calendar import month
import dis
from _pytest import monkeypatch
from _pytest.junitxml import merge_family
import streamlit as st
import json
import datetime
import json
import pandas as pd
import numpy as np
import pickle
from joblib import dump, load

# All fields:
# Vehicle_brand, Vehicle_model, Vehicle_generation, Production_year, Mileage_km, Power_HP, Displacement_cm3, Fuel_type, Drive, Transmission, Type, Doors_number, Colour, Origin_country, First_owner, Province, Month, Day, Year

required_fields = ["Vehicle_brand", "Vehicle_model", "Vehicle_generation", "Mileage_km"]
available_fields = ["Transmission", "Doors_number", "Colour", "First_owner"]
neeeded = ["Fuel_type", 'Condition', 'Production_year', 'Millage', 'Power_HP', 'Displacement_cm3', 'Fuel_type', 'Drive', 'Transmission', 'Type', 'Doors_number', 'Colour', 'Origin_country', 'First_owner', 'Province', 'City', 'Features']
features = ['ABS', 'Electricfrontwindows', 'Driversairbag', 'Powersteering', 'ASR(tractioncontrol)', 'Rearviewcamera', 'Heatedsidemirrors', 'CD', 'Electricallyadjustablemirrors', 'Passengersairbag', 'Alarm', 'Bluetooth', 'Automaticairconditioning', 'Airbagprotectingtheknees', 'Centrallocking', 'Immobilizer', 'Factoryradio', 'Alloywheels', 'Rainsensor', 'On-boardcomputer', 'Multifunctionsteeringwheel', 'AUXsocket', 'Xenonlights', 'USBsocket', 'MP3', 'ESP(stabilizationofthetrack)', 'Frontsideairbags', 'Rearparkingsensors', 'Isofix', 'Aircurtains', 'Tintedwindows', 'Daytimerunninglights', 'Rearsideairbags', 'Foglights', 'Twilightsensor', 'GPSnavigation', 'LEDlights', 'Manualairconditioning', 'Start-Stopsystem', 'Electrochromicrearviewmirror', 'Velorupholstery', 'Electrochromicsidemirrors', 'SDsocket', 'Dualzoneairconditioning', 'Adjustablesuspension', 'Panoramicroof', 'Sunroof', 'Frontparkingsensors', 'Heatedfrontseats', 'Leatherupholstery', 'Electricallyadjustableseats', 'Cruisecontrol', 'Parkingassistant', 'Speedlimiter', 'Heatedwindscreen', 'Electricrearwindows', 'Blindspotsensor', 'Shiftpaddles', 'Aftermarketradio', 'DVDplayer', 'CDchanger', 'Auxiliaryheating', 'Heatedrearseats', 'Four-zoneairconditioning', 'TVtuner', 'Roofrails', 'Activecruisecontrol', 'Hook', 'Laneassistant', 'HUD(head-updisplay)']
fuel_types = ['Gasoline', 'Diesel', 'Gasoline + LPG', 'Hybrid', 'Gasoline + CNG', 'Hydrogen', 'Ethanol']

with open(r'C:\Users\Kacper\Desktop\uczelnia\sem6\Praca\data\columns_df3_dum.json', 'r') as f:
            columns_df3 = json.load(f)
# Load dictionary data
def dictionary_read(file_path):
    with open(file_path, 'r') as file:
        dictionary = json.load(file)
    return dictionary

# Generate list of years from current year to 1900
def generate_years_list():
    current_year = datetime.datetime.now().year
    years_list = list(range(current_year, 1899, -1))
    return years_list

def filter_columns_by_prefix(prefix, columns, x):
    filtered_columns = []
    for col in columns:
        if col.startswith(prefix):
            if x and col.startswith(prefix + x):
                filtered_columns.append(1)
            else:
                filtered_columns.append(0)
    return filtered_columns
# Main function
def main():
    
    with open(r'C:\Users\Kacper\Desktop\uczelnia\sem6\Praca\models\75random_forest_to_app.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Required fields
    st.title("Check your car value!")
# Required fields:
    
    # Load car data dictionary
    car_data_file_path = r"C:\Users\Kacper\Desktop\uczelnia\sem6\Praca\app\dictionary.json"
    car_data = dictionary_read(car_data_file_path)
    

    st.header("Main informations")

    # Brand selection
    brand = st.selectbox('Select Brand', list(car_data.keys()))

    # Model selection based on the brand
    if brand in car_data:
        model = st.selectbox('Select Model', list(car_data[brand].keys()))

    # Version selection based on the brand and model
    if brand in car_data and model in car_data[brand]:
        version = st.selectbox('Select Version', car_data[brand][model])
                
    fuel_type = st.selectbox('Select Fuel Type', fuel_types, key="fuel_type")
    

    # Additional fields
    # st.header("Additional fields:")
    production_year = None  # Initialize production_year variable
    for field in neeeded:
        if field not in required_fields:
            
            if field == 'Condition':
                st.header("Additional informations")
                condition = st.selectbox('Select Condition', ['Used', 'New'])
            elif field == 'Displacement_cm3':
                displacement_cm3 = st.number_input('Select Displacement of engine', min_value=300, max_value=10000, step=1, key='displacement_cm3')
            elif field == 'Production_year':
                max_year = datetime.datetime.now().year
                production_year = st.selectbox('Select Production Year', generate_years_list(), index=0, format_func=lambda x: 'Year' if x == 0 else x, help="Choose production year of the vehicle, limited to the current year.")
            elif field == 'Drive':
                drive_types = ['Front wheels', 'Rear wheels', '4x4 (permanent)', '4x4 (attached automatically)', '4x4 (attached manually)']
                drive = st.selectbox('Select Drive', drive_types)
            elif field == 'Transmission':
                transmission_types = ['Manual', 'Automatic']
                transmission = st.selectbox('Select Transmission', transmission_types)
            elif field == 'Type':
                vehicle_types = ['SUV', 'station_wagon', 'sedan', 'compact', 'city_cars', 'minivan', 'coupe', 'small_cars', 'convertible']
                vehicle_type = st.selectbox('Select Type', vehicle_types)
            elif field == 'Power_HP':
                horse_power = st.number_input('Select Horse Power', min_value=30, max_value=500, step=1, key='power_hp')
            elif field == 'Doors_number':
                doors_number = st.number_input('Select Doors Number', min_value=2, max_value=10, step=1, key='doors_number')
            elif field == 'Colour':
                colours = ['black', 'gray', 'silver', 'white', 'blue', 'other', 'red', 'brown', 'green', 'burgundy', 'golden', 'beige', 'yellow', 'violet']
                colour = st.selectbox('Select Colour', colours)
            elif field == 'First_owner':
                first_owner = st.selectbox('Select First Owner', ['No', 'Yes'])           
            elif field == 'Origin_country':
                st.header("Location informations")
                origin_countries = ['Brak danych', 'Poland', 'Germany', 'France', 'United States', 'Belgium', 'Switzerland', 'Netherlands', 'Italy', 'Austria', 'Sweden', 'Denmark', 'Canada']
                origin_country = st.selectbox('Select Origin Country', origin_countries)
            elif field == 'Province':
                provinces = ['Mazowieckie', 'Opolskie', 'Slaskie', 'Malopolskie', 'Pomorskie', 'Dolnoslaskie', 'Lodzkie', 'Kujawsko-pomorskie', 'Lubelskie', 'Podkarpackie', 'Lubuskie', 'Swietokrzyskie', 'Warminsko-mazurskie', 'Podlaskie', 'Zachodniopomorskie', 'Wielkopolskie', 'Nie znaleziono']
                province = st.selectbox('Select Province', provinces)
            elif field == 'City':
                with open(r'C:\Users\Kacper\Desktop\uczelnia\sem6\Praca\data\cities.json', 'r') as f:
                    cities = json.load(f)
                city = st.selectbox('Select City', cities)
            elif field == 'Features':
                selected_features = st.multiselect('Select Features', features)
            elif field == 'Millage':
                millage = st.number_input('Select millage', min_value=0, max_value=2000000, step=1, key='millage')
            

# brand, model, version, 'fuel_type', 'condition', 'horse_power', 'displacement_cm3','drive','type', 'transmission', 'doors_number','colour','origin_country',
# 'first_owner', 'province', 'city', 'production_year', 'mileage_km', 'displacement_cm3','features'       
             

# Dodaj przycisk i funkcje, ktora ma byc wywolana po jego nacisnieciu
    if st.button("Show Data"):
        # Wczytanie listy z pliku JSON
        data = [brand, model, version, fuel_type, condition, horse_power, displacement_cm3, drive, vehicle_type, transmission, doors_number, colour, origin_country, first_owner, province, city, production_year, millage, displacement_cm3, selected_features]
        selected_indices = [features.index(feature) if feature in selected_features else -1 for feature in features]
        selected_indices = selected_indices + [-1] * (len(features) - len(selected_indices))
        selected_indices = [1 if idx >= 0 else 0 for idx in selected_indices]
        car = [production_year, millage, horse_power, displacement_cm3, doors_number, datetime.datetime.now().month, datetime.datetime.now().day, 
              datetime.datetime.now().year]
        car += selected_indices
        # Wyszukiwanie pol rozpoczynajacych sie od "Vehicle_brand_"
        # poprawic ponizsze, najlepiej zrobic do tego funkcje
        columns_df3_brand = filter_columns_by_prefix("Vehicle_brand_", columns_df3, brand)
        columns_df3_model = filter_columns_by_prefix("Vehicle_model_", columns_df3, model)
        columns_df3_gen = filter_columns_by_prefix("Vehicle_generation_", columns_df3, version)
        columns_df3_fuel = filter_columns_by_prefix("Fuel_type_", columns_df3, fuel_type)
        columns_df3_drive = filter_columns_by_prefix("Drive_", columns_df3, drive)
        columns_df3_transmission = filter_columns_by_prefix("Transmission_", columns_df3, transmission)
        columns_df3_type = filter_columns_by_prefix("Type_", columns_df3, vehicle_type)
        columns_df3_colour = filter_columns_by_prefix("Colour_", columns_df3, colour)
        columns_df3_country = filter_columns_by_prefix("Origin_country_", columns_df3, origin_country)
        columns_df3_first_owner = filter_columns_by_prefix("First_owner_", columns_df3, first_owner)
        columns_df3_province = filter_columns_by_prefix("Province_", columns_df3, province)
        columns_df3_city = filter_columns_by_prefix("City_", columns_df3, city)

        car = car + columns_df3_brand + columns_df3_model + columns_df3_gen + columns_df3_fuel + columns_df3_drive + columns_df3_transmission + columns_df3_type + columns_df3_colour + columns_df3_country + columns_df3_first_owner + columns_df3_first_owner + columns_df3_province + columns_df3_city
        # lista car posiada wypelnione pola to pol features, zostaly jeszcze kategoryczne do dokonczenia
        car_array = np.array(car)
        car_reshaped = car_array.reshape(1, -1)
        df = pd.DataFrame(car_reshaped, index=[0], columns=columns_df3)
        st.write(df)
        # Scaler!!!!
        scaler = load(r'C:\Users\Kacper\Desktop\uczelnia\sem6\Praca\models\standard_scaler.joblib')
        scaled = scaler.transform(df.iloc[[0]].values)
        pred = loaded_model.predict(scaled)
        st.write(pred)
    # Wczytanie listy z pliku JSON
    with open(r'C:\Users\Kacper\Desktop\uczelnia\sem6\Praca\data\columns_df3_dum.json', 'r') as f:
        columns_df3_dum2 = json.load(f)
if __name__ == "__main__":
    main()

