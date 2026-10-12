from nimbus.support.desk import customer_profile, reply_language, system_prompt, unbacked_claims
from nimbus.support.tools import SupportStore


def test_reply_language_is_decided_from_the_message():
  assert reply_language("What's on my subscription right now?") == "English"
  assert reply_language("There were insects in my coffee bag!") == "English"
  assert reply_language("Bhai mera subscription 20 hafte ke liye rok do") != "English"
  assert reply_language("मेरा ऑर्डर NC-20981 अभी तक नहीं आया") != "English"


def test_system_prompt_names_the_language_and_the_subscription():
  store = SupportStore()
  prompt = system_prompt(customer_profile(store, "c-5005"), "English")
  assert "in English" in prompt and "customer's language" not in prompt
  assert "has no bean subscription" in prompt
  assert "has a bean subscription" in system_prompt(customer_profile(store, "c-1042"))


def test_unbacked_claims_need_the_matching_action():
  reply = "Your subscription has been paused for 20 weeks."
  assert unbacked_claims(reply, []) == ["reply claims pause_subscription but it never ran"]
  assert unbacked_claims(reply, ["pause_subscription(4 weeks)"]) == []
  assert unbacked_claims("I’ve escalated this urgently.", []) == ["reply claims escalate but it never ran"]
  assert unbacked_claims("Subscriptions can be paused for 1 to 12 weeks.", []) == []
