import streamlit as st
import joblib
import numpy as np
import time

def loadPickle(filename):
    '''
    Load pickle file

    Args:
        filename: filename of pickle file

    Returns:
        loaded pickle file
    '''
    with open(filename, 'rb') as file:
        return joblib.load(file)

model = loadPickle('model/xgb_model.pkl')
le_market_segment = loadPickle('labelEncoder/le_market_segment.pkl')
le_meal_plan = loadPickle('labelEncoder/le_meal_plan.pkl')
le_room_reserved = loadPickle('labelEncoder/le_room_reserved.pkl')
le_booking_status = loadPickle('labelEncoder/le_booking_status.pkl')

def predict_with_model(model, data):
  '''
  Predict with model

  Args:
    model: model to predict
    data: {List of Data} data to predict

  Returns:
    prediction
  '''
  data[13] = le_meal_plan.transform([data[13]])[0]
  data[14] = le_room_reserved.transform([data[14]])[0]
  data[15] = le_market_segment.transform([data[15]])[0]  

  print(data)
  time_start = time.time()
  predictions = model.predict([data])
  time_end = time.time()
  st.text(f'Prediction took: {np.round(time_end - time_start, 4)} sec')
  return le_booking_status.inverse_transform(predictions)[0]

def main():
    st.title('Will This Booking got Canceled')
    st.subheader('Imerson Sanmarlow Krysthio - 2702281260 UTS MD')

    #User Input
    no_of_adults = st.number_input("No of Adults", min_value = 0, max_value = 4)
    no_of_children = st.number_input("No of Children", min_value = 0, max_value = 10)
    no_of_weekend_nights = st.number_input("No of Weekend Nights", min_value = 0, max_value = 7)
    no_of_week_nights = st.number_input("No of Week Nights", min_value = 0, max_value = 17)

    type_of_meal_plan = st.selectbox("Type of Meal Plan", ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])

    ##User Input {Yes/No}
    opt = {'Yes': 1, 'No': 0}
    rcps_choice = st.radio("Required Car Parking Space?", list(opt.keys()))
    required_car_parking_space = opt[rcps_choice]

    room_type_reserved = st.selectbox("Type of Room Reserved", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])

    lead_time = st.number_input("Lead Time", min_value = 0, max_value = 443)
    arrival_month = st.number_input("Arrival Month", min_value = 1, max_value = 12)
    arrival_date = st.number_input("Arrival Date", min_value = 1, max_value = 31)

    market_segment_type = st.selectbox("Market Segment Type", ['Offline', 'Online', 'Corporate', 'Complementary', 'Aviation'])
    
    ##User Input {Yes/No}
    rg_choice = st.radio("Is Repeated Guest?", list(opt.keys()))
    repeated_guest = opt[rg_choice]

    no_of_previous_cancellations = st.number_input("No of Previous Cancelations", min_value = 0, max_value =13)
    no_of_previous_bookings_not_canceled = st.number_input("No of Previous Bookings Not Canceled", min_value = 0, max_value = 58)

    avg_price_per_room = st.number_input("Avg Price per Room (Euro)", min_value = 0.0, max_value = 365.0)

    no_of_special_requests = st.number_input("No of Special Requests", min_value = 0, max_value = 5)

    if st.button('Make Decision!'):
        data = [no_of_adults, no_of_children, no_of_weekend_nights, 
                no_of_week_nights, required_car_parking_space, lead_time, 
                arrival_month, arrival_date, repeated_guest, no_of_previous_cancellations, 
                no_of_previous_bookings_not_canceled, avg_price_per_room, 
                no_of_special_requests, type_of_meal_plan, room_type_reserved, 
                market_segment_type]

        st.success(f'This Customer Would Likely {predict_with_model(model, data).replace("_", " ")}')

    


if __name__ == '__main__':
    main() 