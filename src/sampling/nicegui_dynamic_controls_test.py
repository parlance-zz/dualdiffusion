from nicegui import ui

# Initial data source (dictionary)
data = {
    'Slider 1': 50,
    'Input 1': 'Hello',
}

def handle_change(control_name, event_value):
    # Update the dictionary with the new value
    data[control_name] = event_value
    print(f"{control_name} changed to: {event_value}")

def create_controls():
    # Clear previous controls
    container.clear()

    # Iterate over the dictionary to create controls
    for label, value in data.items():
        if isinstance(value, bool):
            # Create a checkbox for boolean values
            control = ui.checkbox(label=label, value=value)
        elif isinstance(value, str):
            # Create an input for string values
            control = ui.input(label=label, value=value)
        elif isinstance(value, (int, float)):
            # Create a slider for numeric values
            control = ui.slider(value=value, min=0, max=100)
        
        # Attach the event handler for the control
        control.on('update', lambda e, control=control: handle_change(label, e.value))

# Create a container for the controls
with ui.column() as container:
    create_controls()  # Initially create controls

# Example button to add a new control dynamically
def add_control():
    new_label = f'Input {len(data) + 1}'
    data[new_label] = 'New Value'  # Add a new item to the dictionary
    create_controls()  # Refresh controls based on updated dictionary

# Button to add a control
ui.button('Add Control', on_click=add_control)

ui.run()
