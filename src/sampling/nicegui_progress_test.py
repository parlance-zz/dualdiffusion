from nicegui import ui

# Create a container with relative positioning for the image and progress bar overlay
with ui.card().classes('relative w-64 h-64'):
    with ui.image('https://picsum.photos/400/400').classes('absolute top-0 left-0 w-full h-full') as img:

        # Create the linear progress bar and position it on top of the image
        with ui.linear_progress(value=0., show_value=True).classes('absolute top-50 left-0 w-full z-1') as progress:
            
            #progress_label = ui.label('0%').classes('text-subtitle2 text-center text-white items-center z-0')
            #absolute-center text-sm text-white
            p = 0

            def progress_update():
                global p
                p = min(p + 0.01, 1)
                progress.set_value(f"{int(p*100)}%")
                if p >= 1:
                    progress.set_visibility(False)
                    
                #progress_label.set_text(f'{int(progress.value * 100)}%')

            ui.timer(0.1, progress_update)

ui.run()