# %%
import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
DATA_PATH = "apartments_data_enriched_with_new_features.csv"
df = pd.read_csv(DATA_PATH)
df = df.dropna().drop_duplicates()

features = ['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 'room_per_m2', 'luxurious', 'temporary', 'furnished', 'area_cat_ecoded', 'zurich_city', 'avg_price_postal_rooms_area']

X = df[features]
y = df['price']

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

def predict_price(rooms, area, pop, pop_dens, frg_pct, emp, tax_income, room_per_m2, luxurious, temporary, furnished, area_cat_ecoded, zurich_city, avg_price_postal_rooms_area):
    input_df = pd.DataFrame({
        'rooms': [rooms],
        'area': [area],
        'pop': [pop],
        'pop_dens': [pop_dens],
        'frg_pct': [frg_pct],
        'emp': [emp],
        'tax_income': [tax_income],
        'room_per_m2': [room_per_m2],
        'luxurious': [luxurious],
        'temporary': [temporary],
        'furnished': [furnished],
        'area_cat_ecoded': [area_cat_ecoded],
        'zurich_city': [zurich_city],
        'avg_price_postal_rooms_area': [avg_price_postal_rooms_area]
    })
    price = model.predict(input_df)[0]
    return round(price, 2)

inputs = [
    gr.Number(label="Rooms"),
    gr.Number(label="Area (m2)"),
    gr.Number(label="Population"),
    gr.Number(label="Population Density"),
    gr.Number(label="% Foreigners"),
    gr.Number(label="Employees"),
    gr.Number(label="Tax Income"),
    gr.Number(label="Room per m2"),
    gr.Number(label="Luxurious (0/1)"),
    gr.Number(label="Temporary (0/1)"),
    gr.Number(label="Furnished (0/1)"),
    gr.Number(label="Area Category Encoded"),
    gr.Number(label="Zurich City (0/1)"),
    gr.Number(label="Avg Price Postal Rooms Area")
]

iface = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=gr.Number(label="Predicted Price"),
    title="Apartment Price Prediction",
    description="Enter apartment features to predict the price."
)

if __name__ == "__main__":
    iface.launch()

# %%
