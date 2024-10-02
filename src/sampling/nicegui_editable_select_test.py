from nicegui import ui

select_options = ["Google", "Facebook", "Twitter", "Apple", "Oracle"]

with ui.row():
    select_element = ui.select(
        options=select_options,
        label="Type or select an option",
        value="Google",
        with_input=True,
    )

select_val = "Google"
def on_input_value(val):
    print("filter", f"'{val}'")
    global select_val
    select_val = val

def on_blur(e):
    print("blur", "select_val:", select_val, "select_element.value:", select_element.value)  
    if select_val != "" and select_val not in select_options:
        select_element.options = select_options + [select_val]
        select_element.set_value(select_val)

def on_value_change(val):
    print("value_change", val)
    if val in select_options:
        select_element.set_options(select_options)

select_element.on("input-value", lambda e: on_input_value(e.args))
select_element.on("blur", lambda e: on_blur(e))
select_element.on_value_change(lambda e: on_value_change(e.value))

ui.run()