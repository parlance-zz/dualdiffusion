from nicegui import ui

drawer = ui.drawer("right").classes('w-64')  # You can adjust width

with ui.row():
    

    with drawer:
        ui.label('Side panel content here')

    # Add a button to toggle the side panel visibility
    toggle_button = ui.button('Toggle side panel', on_click=lambda: drawer.toggle())

    with ui.column():
        ui.label('Main content area')
        ui.button('Another button')

ui.run()