from __future__ import annotations

import os

import httpx
import streamlit as st


API_URL = os.getenv("SERVICE_AGENT_API_URL", "http://127.0.0.1:8000")
DEMO_IDENTITIES = {
    "Alex Morgan — CUST-1001, CUST-1002": "demo-agent-alex",
    "Jordan Lee — CUST-1003 only": "demo-agent-jordan",
}

st.set_page_config(page_title="FDE Servicing Agent", page_icon="💬")
st.title("FDE Servicing Agent")
st.caption("Milestone 1 · local synthetic CRM data · no Salesforce connection")

identity_label = st.sidebar.selectbox("Demo user", list(DEMO_IDENTITIES))
demo_user = DEMO_IDENTITIES[identity_label]
st.sidebar.info(
    "Permissions are resolved by the API from this demo identity. "
    "Text entered in chat cannot change them."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Try: What is the status of CASE-4821?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a service case"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = httpx.post(
                f"{API_URL}/chat",
                headers={"X-Demo-User": demo_user},
                json={"message": prompt},
                timeout=15.0,
            )
            response.raise_for_status()
            result = response.json()
            reply = result["reply"]
            st.markdown(reply)
            with st.expander("Execution details"):
                st.json(
                    {
                        "execution_id": result["execution_id"],
                        "intent": result["intent"],
                        "steps": result["steps"],
                    }
                )
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            reply = (
                "The local API is unavailable or returned an invalid response. "
                f"Start it at {API_URL} and try again. ({exc})"
            )
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

