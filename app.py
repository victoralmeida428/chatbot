import mesop as me
import mesop.labs as mel
from ChatLLM import generate_answer

@me.stateclass
class State:
  data: str
  is_loading: bool

@me.page(
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://google.github.io"]
  ),
  path="/chat",
  title="Mesop Demo Chat",
)
def page():
  state = me.state(State)
  if state.is_loading:
    me.progress_spinner()
  mel.chat(transform, title="Controll Chat", bot_user="Sr. Márcio")
  


def transform(input: str, history: list[mel.ChatMessage]):
    try:
        state = me.state(State)
        state.is_loading=True
        answer = generate_answer(input, history)
        state.is_loading=False
         
        yield answer
    except NameError as e:
        yield 'Error'
        
    

