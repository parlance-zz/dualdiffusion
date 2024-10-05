from nicegui import ui

# Function to handle star rating selection
def set_rating(rating):
    for i in range(0, 5):
        star = stars[i]
        if i <= rating:
            star.style('color: gold;')  # Highlight selected stars
        else:
            star.style('color: lightgray;')  # Default color for unselected stars

# Create a container for the rating stars
stars = []
with ui.row():
    for i in range(0, 5):
        star = ui.icon('star').style('cursor: pointer;').on('click', lambda _, i=i: set_rating(i))
        stars.append(star)

# Start the NiceGUI app
ui.run()