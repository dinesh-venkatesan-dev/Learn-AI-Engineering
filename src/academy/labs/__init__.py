from nimbus.config import get_settings
from nimbus.observability.tracing import guard_uploads

guard_uploads(get_settings())  # before any lab creates a LangSmith client: no Nimbus key → nothing leaves this machine
