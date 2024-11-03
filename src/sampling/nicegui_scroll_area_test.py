from nicegui import ui

#with ui.scroll_area():
#    ui.image('https://picsum.photos/2000/500').style('height: 300px;')
with ui.scroll_area().classes('bg-blue-100 w-40 h-20'):
    with ui.row(wrap=False):
        ui.button('A')
        ui.button('B')
        ui.button('C')
        ui.button('D')

ui.run()