from nicegui import ui

class ReorderableList:
    def __init__(self, items):
        self.items = items
        self.render_list()

    def render_list(self):
        """Render the entire list with buttons to reorder."""
        ui.column().clear()
        with ui.column().classes('gap-4'):
            for i, item in enumerate(self.items):
                with ui.row().classes('items-center gap-2'):
                    ui.label(item).classes('w-32')
                    
                    ui.button('▲').on_click(lambda i=i: self.move_up(i)).classes('bg-blue-500 text-white rounded-lg px-2 py-1').set_enabled(i > 0)
                    ui.button('▼').on_click(lambda i=i: self.move_down(i)).classes('bg-blue-500 text-white rounded-lg px-2 py-1').set_enabled(i < len(self.items) - 1)

    def move_up(self, index):
        """Move the item up the list."""
        if index > 0:
            self.items[index], self.items[index - 1] = self.items[index - 1], self.items[index]
            self.render_list()

    def move_down(self, index):
        """Move the item down the list."""
        if index < len(self.items) - 1:
            self.items[index], self.items[index + 1] = self.items[index + 1], self.items[index]
            self.render_list()

# Example usage
items = ['Item 1', 'Item 2', 'Item 3', 'Item 4']
reorderable_list = ReorderableList(items)

ui.run()
