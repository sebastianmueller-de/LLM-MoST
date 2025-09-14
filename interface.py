import time
import gradio as gr

from process_engine import ProcessEngine
from config import SessionConfig

import os
from dotenv import load_dotenv

load_dotenv()
DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", "").strip()
DEFAULT_API_MODEL = os.getenv("DEFAULT_API_MODEL", "").strip()

DEFAULT_OLLAMA_URL = os.getenv("DEFAULT_OLLAMA_URL", "").strip()
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "").strip()

DEFAULT_MODE = os.getenv("DEFAULT_MODE", "").strip()

global_configs = {
    "key": DEFAULT_API_KEY if DEFAULT_API_KEY else None,
    "model": DEFAULT_API_MODEL if DEFAULT_API_MODEL else None,
    "ollama_url": DEFAULT_OLLAMA_URL if DEFAULT_OLLAMA_URL else None,
    "ollama_model": DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL else None,
    "mode": DEFAULT_MODE if DEFAULT_MODE else None
}

class ChatHandler:
    def __init__(self):
        self.engine = ProcessEngine()
        self.step = 1

    def chat(self, user_input:str, image_list, index):
        response = ""
        gifs = []

        self.engine.set_llm(llm_parameters=global_configs)

        self.engine.handle_input(self.step, user_input)

        if self.step == 1: # TODO: immediate fail if location not present in DB
            response = "Please provide a general description of the scenario and its content (e.g., highway situations, merging lanes, traffic jams, …)"

        elif self.step == 2:
            response = "Great! Now, please give a more detailed description of the road on which the ego is driving (traffic signs, traffic lights)" # Missing more complex information like geometry/intersections

        elif self.step == 3:
            response = "Please describe your desired obstacles next. Obstacles include both static (parked cars, construction zones, ...) and dynamic ones (cars, trucks, pedestrians, ...)"

        elif self.step == 4:
            response = "You now have the option of specifying an initial velocity for the ego vehicle."

        elif self.step == 5:
            top_results, file_path = self.engine.end_and_wait()
            if len(top_results) > 0:
                response = f"The conversation is complete. Best matching scenario: {top_results[0]} out of {len(top_results)} final candidates. Press enter on a selected scenario to start testing."
                self.step = 6 # Increase step, so that at the end step 7 is returned -> Enter modification
            else:
                response = f"Sorry, I could not find a fitting scenario. Press the restart button to query another."
                # Returns step 6 at the end, so that user is forced to start a new search

        elif self.step == 6:
            self.step = 5
            response = "Press the restart button to query another scenario."

        # At step 7, the logic changes. After a scenario has been found, the GIF should be displayed and the modification should be entered. This should update the image_list to only contain the GIF (the list is updated in the process engine class).
        elif self.step == 7:
            response = "This is the GIF of the chosen scenario. Let me know if you wish to modify it, or start running the motion planner."

            print("image_list", image_list)
            print("index", index)

            # If the user has not yet looked at a single image, two options are possible: 1) No results were found (should be caught and avoided before step 7 is ever reached) 2) The corresponding GIF was not found
            # 3) The user has simply decided not to look at any images --> perform an extra check here by loading images for the final result and choosing the first one
            if len(image_list) == 0:
                image_list = self.engine.get_images()[0] # get_images() returns a tuple of images, step
                if len(image_list) == 0:
                    raise ValueError("There are no images available.")
                index = 0

            gifs = self.engine.select_initial_gif(image_list[index])

            if gifs is None:
                response = "Sorry, I could not create a GIF for this scenario. Please press the restart button to query another scenario."
                self.step = 6
                images, image_step = self.engine.get_images()

                return response, [], index, 5, images, image_step

            index = 0

        # In this step, we get the input from the user, for what he wants to do. Depending on this, different methods might be called (modification, motion planner execution, etc.)
        # TODO: the logic here changes - we can no longer simply use pre-written dialogue, because LLM output matters --> need to change logic so that from this point onwards, the responses are created by the process engine class
        elif self.step == 8:
            # Supply images and index so that processor knows which one desired
            gifs, index, string_output = self.engine.react_to_request(user_input, image_list, index)
            images, image_step = self.engine.get_images()

            if gifs is None:
                response = "Sorry, something went wrong in the modification process. Please press the restart button to query another scenario."
                self.step = 6

                return response, [], index, 5, images, image_step

            # If relevant LLM output has been generated during the process, it will be displayed instead of a scripted answer
            # Unlike the failure from above, we stay in the modification phase
            if len(string_output) > 0:
                response = string_output
                self.step = 8
                return response, gifs, index, self.step, images, image_step

            # Reset step, so that we remain in a loop for step 8
            self.step = 7

            print("\n\nGIFs of the chosen scenario: ", gifs)

            response = "Here is your modification. Would you like to proceed?"

        self.step += 1

        images, image_step = self.engine.get_images()

        return response, gifs, index, self.step, images, image_step


handler = ChatHandler()

planner_options = list(handler.engine.planners.keys())

def chatbot_stream(user_input: str, history, image_list, index):
    response, gifs, index, step, images, image_step = handler.chat(user_input, image_list, index)
    if not history:
        history = []
    history.append({"role": "user", "content": user_input})

    # Placeholder
    assistant_message = {"role": "assistant", "content": ""}
    history.append(assistant_message)

    # This means that we are not yet in the second phase (no GIFs are loaded in) -> we just update images to the latest ones
    if len(gifs) == 0 and len(images) > 0:
        # Simulate streaming response
        for partial_text in simulated_streaming_response(response):
            assistant_message["content"] = partial_text

            # Why are we returning history twice? --> easiest fix for our format, which allows us to start the chat with a bot message

            # The index should not jump around, if the selected image is not removed from the candidate list between two steps
            # If it is removed, the index should be set to 0 to guarantee being in bounds
            if index >= len(image_list):
                index = 0

            if len(image_list) > 0:
                current_image = image_list[index]
                filename = current_image.split("/")[-1].removesuffix(".png")
                label = f"**{filename}**  {index + 1}/{len(image_list)} candidates after step {image_step}"
            else:
                label = ""

            if image_step == 1:
                selected = "1. Location"
            elif image_step == 2:
                selected = "2. Tags"
            elif image_step == 3:
                selected = "3. Road Network"
            elif image_step == 4:
                selected = "4. Obstacles"
            else:
                selected = "5. Velocity"

            yield history, history, gr.update(value=""), gr.update(value=images[index]), images, index, gr.update(value=label, visible=True), gr.update(value=selected), image_step, gr.update(), gr.update()

    elif len(gifs) == 0 and len(images) == 0:
        # Simulate streaming response
        for partial_text in simulated_streaming_response(response):
            assistant_message["content"] = partial_text

            # Why are we returning history twice? --> easiest fix for our format, which allows us to start the chat with a bot message

            yield history, history, gr.update(value=""), gr.update(), images, 0, gr.update(), gr.update(), step - 1, gr.update(), gr.update()

    else:
        # Simulate streaming response
        for partial_text in simulated_streaming_response(response):
            assistant_message["content"] = partial_text

            filename = gifs[index].split("/")[-1].removesuffix(".gif")
            caption = f"**{filename}**  {index + 1}/{len(gifs)} plots"

            yield history, history, gr.update(value=""), gr.update(value=gifs[index]), gifs, index, caption, gr.update(visible=False), step, gr.update(visible=False), gr.update(value="Run with Motion Planner", visible=True)

# output format of corr. gradio function: [history, chatbot, user_input, image_output, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix]


def simulated_streaming_response(full_text, delay=0.02):
    current_text = ""
    for char in full_text:
        current_text += char
        yield current_text
        time.sleep(delay)

def get_debug_info():
    return handler.engine.get_debug_log()

# Currently it is a little unclean with the order in which the restart happens and debug information is displayed
def restart():
    handler.step = 1
    handler.engine.restart()

    history = [{"role": "assistant", "content": "Hi, welcome to the scenario generation. To start the process, please provide me with a location for the scenario, or let me know if you do not have any specific place in mind."}]

    return history, history, gr.update(value=None, visible=True), gr.update(value=""), [], 0, gr.update(value=""), gr.update(choices=["1. Location", "2. Tags", "3. Road Network", "4. Obstacles", "5. Velocity"], value=None, visible=True), 5, gr.update(value=None, visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False)
# Output order: [history, chatbot, image_output, user_input, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix, planner_output]

# Loads in the newest batch of images
# Returns the new list of file paths, resets the index to 0 and displays the first of the new images
def load_images():

    images, step = handler.engine.get_images()

    if len(images) == 0:
        return images, 0, gr.update(value=None), gr.update(value=None), step, gr.update(value=None)

    filename = images[0].split("/")[-1].removesuffix(".png")
    label = f"**{filename}**  1/{len(images)} candidates after step {step}"

    return images, 0, gr.update(value=images[0]), gr.update(value=label), step, gr.update(value=None,choices=["1. Location", "2. Tags", "3. Road Network", "4. Obstacles", "5. Velocity"])

# Returns the updated index, as well as the  file path of the next image to be displayed
def show_next_image(images, index, step):
    number_images = len(images)

    if number_images == 0:
        return index, None, gr.update(value=None)

    index = (index + 1) % number_images
    current_image = images[index]

    # This means we still are in the query mode
    if step < 7:
        # Extract file name
        filename = current_image.split("/")[-1].removesuffix(".png")
        label = f"**{filename}**  {index + 1}/{number_images} candidates after step {step}"
    # Else we are in the GIF mode and no longer should consider candidates & steps
    else:
        filename = current_image.split("/")[-1].removesuffix(".gif")
        label = f"**{filename}**  {index + 1}/{number_images} plots"

    return index, current_image, gr.update(value=label, visible=True)

def show_previous_image(images, index, step):
    number_images = len(images)

    if number_images == 0:
        return index, None, gr.update(value=None)

    index = index - 1 if index > 0 else number_images - 1
    current_image = images[index]

    if step < 7:
        filename = current_image.split("/")[-1].removesuffix(".png")
        label = f"**{filename}**  {index + 1}/{number_images} candidates after step {step}"
    else:
        filename = current_image.split("/")[-1].removesuffix(".gif")
        label = f"**{filename}**  {index + 1}/{number_images} plots"

    return index, current_image, gr.update(value=label, visible=True)

# TODO: there is some bug with the step selector and the input from above
def select_step(selected_step):
    if selected_step == "1. Location":
        step = 1
    elif selected_step == "2. Tags":
        step = 2
    elif selected_step == "3. Road Network":
        step = 3
    elif selected_step == "4. Obstacles":
        step = 4
    elif selected_step == "5. Velocity":
        step = 5
    else:
        step = 7
    images, step = handler.engine.get_selected_images(step)

    print(images)

    if len(images) == 0:
        return images, 0, gr.update(value=None), gr.update(value=None), step

    filename = images[0].split("/")[-1].removesuffix(".png")
    label = f"**{filename}**  1/{len(images)} candidates after step {step}"

    return images, 0, gr.update(value=images[0]), gr.update(value=label), step

def run_frenetix_planner(gifs, index):
    planner_gifs, planner_index = handler.engine.run_with_planner(gifs, index)
    return gr.update(value=planner_gifs[planner_index], visible=True)
# Return order: [planner_output]

# Method that takes in clicks on images and processes them
# Used to map a goal area chosen by a user to the coordinates of the CR plot
def image_click(image_list, index, event: gr.SelectData):
    px, py = event.index  # pixel click

    # Helper function that transforms a clicked pixel into the dimensions from the CR plot
    def pixel_to_world(px, py, plot_limits, img_width, img_height):
        x_min, x_max, y_min, y_max = plot_limits
        x_world = x_min + px / img_width * (x_max - x_min)
        y_world = y_max - py / img_height * (y_max - y_min)
        return x_world, y_world

    # Load in the information from the JSON object storing the specific dimensions of a graph
    metadata = handler.engine.load_plot_dimensions_json(image_list[index])

    # The mathematical formula for this is:
    # x = x_min + px / img_width * (x_max - x_min)
    # y = y_max - py / img_height * (y_max - y_min)
    img_w, img_h = int(metadata["dpi"] * metadata["figsize"][0]), int(metadata["dpi"] * metadata["figsize"][1])
    coords = pixel_to_world(px, py, metadata["plot_limits"], img_w, img_h)

    print(f"\n SELECTED COORDS IN CR DIMENSIONS: x={coords[0]}, y={coords[1]}\n\n")

# Allows the user to activate the goal-region choice mode
def toggle_interactivity(current_state):
    return gr.update(interactive=not current_state), not current_state

def poll_console_output():
    return handler.engine.console_log

def poll_plot():
    return handler.engine.plot

# Sets the motion planner from the engine to the planner selected in the Radio
def select_motion_planner(planner: str):
    handler.engine.set_planner(planner)

def save_settings(api_or_url, model_name, mode_choice):
    # Mode: always take what the user selected (Commercial/Ollama)
    global_configs["mode"] = mode_choice if mode_choice else global_configs.get("mode")

    # API key / URL: prefer user input, otherwise keep existing (default from .env)
    if api_or_url.strip():
        global_configs["key"] = api_or_url.strip()
    elif not global_configs.get("key"):
        # no user input and no default from .env
        return (
            "❌ No API key or URL found. Please enter one.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # Model: prefer user input, otherwise keep existing (default from .env)
    if model_name.strip():
        global_configs["model"] = model_name.strip()

    # If we're in Ollama mode, map correctly
    if global_configs["mode"] == "Ollama":
        if api_or_url.strip():
            global_configs["ollama_url"] = api_or_url.strip()
        elif not global_configs.get("ollama_url"):
            return (
                "❌ No Ollama URL found. Please enter one.",
                gr.update(visible=False),
                gr.update(visible=False),
            )

        if model_name.strip():
            global_configs["ollama_model"] = model_name.strip()

        return (
            f"✅ Ollama mode set! URL: {global_configs['ollama_url']} | Model: {global_configs['ollama_model'] or '(default)'}",
            gr.update(visible=True),
            gr.update(visible=True),
        )

    # Otherwise, Commercial mode
    return (
        f"✅ Commercial mode set! Key: {'(using default)' if not api_or_url.strip() else '(custom)'} | Model: {global_configs['model'] or '(default)'}",
        gr.update(visible=True),
        gr.update(visible=True),
    )


with gr.Blocks(theme=gr.themes.Soft(), css=".debug-output { font-family: monospace; background-color: #f9f9f9; border: 1px solid #ccc; padding: 8px; }") as demo:
    with gr.Tab("Setup"):
        api_input = gr.Textbox(label="API Key (or local URL if using Ollama)", type="password")
        model_input = gr.Textbox(label="Model (either a Gemini-based model or a local model)", type="text")
        commercial_or_os = gr.Radio(choices=["Commercial", "Ollama"], value="Commercial")
        confirm_btn = gr.Button("Confirm")
        status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Main App"):
        history = gr.State(value = [ {"role": "assistant", "content": "Hi, welcome to the scenario generation. To start the process, please provide me with a location for the scenario, or let me know if you do not have any specific place in mind."} ])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(history.value, type="messages", height=700, visible=False)
                user_input = gr.Textbox(label="Your Message", visible=False)
            with gr.Column(scale=1):
                # Debug output field
                debug_output = gr.Textbox(label="Debug Output", lines=38, interactive=False, every=1, value=get_debug_info, scale=1, elem_classes=["debug-output"], show_copy_button=True)
        with gr.Row():
            send_btn = gr.Button("Send", scale=3)

            # Restart Logic, remove if bad
            restart_btn = gr.Button("Restart", scale=1)

        image_list_state = gr.State([])
        index_state = gr.State(0)
        selected_step_state = gr.State(5)

        # TODO: currently invisible --> still needs to be clicked twice, if step selector has been used before
        image_update = gr.Button("Update Images", visible=False)

        image_output = gr.Image(type="filepath", interactive=False, visible=True)
        image_output.select(fn=image_click, inputs=[image_list_state, index_state], outputs=[])

        image_caption = gr.Markdown(value="", visible=True)

        with gr.Row():
            image_previous = gr.Button("< Previous", scale=1)
            image_next = gr.Button("Next >", scale=1)

        # The logic here is this: image_update should always override the step selection and show the latest images
        # The step_selector is there for a comparison with the previous steps
        step_selector = gr.Radio(["1. Location", "2. Tags", "3. Road Network", "4. Obstacles", "5. Velocity"], label="Select Step")
        step_selector.change(fn=select_step, inputs=[step_selector], outputs=[image_list_state, index_state, image_output, image_caption, selected_step_state])

        # Button that becomes visible as soon as the first GIF is chosen and runs the selected scenario with Frenetix
        run_frenetix = gr.Button("Run with Motion Planner", scale=1, visible=False)

        # TODO
        select_planner = gr.Radio(choices=planner_options, label="Select Planner")
        select_planner.change(fn=select_motion_planner, inputs=[select_planner], outputs=[])

        # Output for the scenarios run through a planner
        planner_output = gr.Image(type="filepath", visible=False)

        with gr.Row():
            console_output = gr.Textbox(
                label="Console Output",
                lines=20,
                interactive=False,
                every=1,
                value=poll_console_output,
                scale=1,
                elem_classes=["debug-output"],
                show_copy_button=True
            )
        with gr.Row():
            plot = gr.Plot(label="Batch Simulation Stats", every=1, value=poll_plot)

        # Button that lets the user update images to the latest search result
        # Also resets the step selector to not have anything selected
        # TODO: has to be clicked twice, before it overrides the selected step --> why is this the case? --> Probably same issue as with Restart button - we need to set a value in the gr.update()
        image_update.click(fn=load_images, inputs=[], outputs=[image_list_state, index_state, image_output, image_caption, selected_step_state, step_selector])

        # Buttons to click through the images
        image_next.click(fn=show_next_image, inputs=[image_list_state, index_state, selected_step_state], outputs=[index_state, image_output, image_caption])
        image_previous.click(fn=show_previous_image, inputs=[image_list_state, index_state, selected_step_state], outputs=[index_state, image_output, image_caption])


        send_btn.click(chatbot_stream, [user_input, history, image_list_state, index_state], [history, chatbot, user_input, image_output, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix])
        user_input.submit(chatbot_stream, [user_input, history, image_list_state, index_state], [history, chatbot, user_input, image_output, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix])

        # Only allows clean restart, if chatbot is currently not writing text (some kind of yield/streaming conflict)
        restart_btn.click(restart, inputs=[], outputs=[history, chatbot, image_output, user_input, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix, planner_output])

        run_frenetix.click(fn=run_frenetix_planner, inputs=[image_list_state, index_state], outputs=[planner_output])

    confirm_btn.click(
        save_settings,
        [api_input, model_input, commercial_or_os],  # inputs
        [status, chatbot, user_input]  # outputs
    )


# demo.launch()

def prompt_user_for_bool():
    while True:
        answer = input("Use the scenario repair module this session? (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        elif answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")

if __name__ == "__main__":


    use_special = prompt_user_for_bool()
    SessionConfig.set_use_repair_module(use_special)

    demo.launch()