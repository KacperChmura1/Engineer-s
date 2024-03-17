import streamlit as st
import csv

# All fields:
# 	Vehicle_brand	Vehicle_model	Vehicle_generation	Production_year	Mileage_km	Power_HP	Displacement_cm3	Fuel_type	Drive
# 	Transmission	Type	Doors_number	Colour	Origin_country	First_owner	Province	Month	Day	Year

required_fields = ["Vehicle_brand", "Vehicle_model", "Vehicle_generation", "Production_year", "Mileage_km", "Power_HP", "Displacement_cm3",
                   "Fuel_type"]
available_fields =  ["Transmission", "Doors_number", "Colour", "First_owner"]

import json

def dictionary_read(nazwa_pliku):
    with open(nazwa_pliku, 'r') as plik:
        dictionary = json.load(plik)
    return dictionary

nazwa_pliku = r"C:\Users\Kacper\Desktop\uczelnia\sem6\Praca\app\dictionary.json"
car_data = dictionary_read(nazwa_pliku)


# Dictionary of constraints

def main():
    # Required fields
    st.title("Check your car value!")
    st.header("Required fields:")
    # Brand selection
    brand = st.selectbox('Select Brand', list(car_data.keys()))

    # Model selection based on the brand
    if brand in car_data:
        model = st.selectbox('Select Model', list(car_data[brand].keys()))

    # Version selection based on the brand and model
    if brand in car_data and model in car_data[brand]:
        version = st.selectbox('Select Version', car_data[brand][model])




if __name__ == "__main__":
    main()
