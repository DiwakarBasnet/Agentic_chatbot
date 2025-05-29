# Agentic Chatbot 🤖

## Features

* **Agentic Architecture**: Modular agents (`agents.py`) manage specific tasks, enabling scalable and maintainable code.
* **Retrieval-Augmented Generation (RAG)**: Utilizes `rag_util.py` to fetch relevant information, enhancing response accuracy.
* **Calendar Integration**: Automates scheduling tasks via `calendar_invite.py`, streamlining event management.

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* [Conda](https://docs.conda.io/en/latest/) (for environment management)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DiwakarBasnet/Agentic_chatbot.git
   cd Agentic_chatbot
   ```

2. **Create and Activate the Conda Environment**

   ```bash
   conda env create -f environment.yml
   conda activate chatbot
   ```

3. **Install Additional Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**

   ```bash
   streamlit run streamlit_app.py
   ```

Awesome question — you should definitely note that clearly in your `README.md` so users know how to set up the necessary credentials before running your app.

You can add a section under **Getting Started** or create a new **🔐 Environment Variables & API Keys** section.
Here’s how you can write it:

---

## 🔐 Environment Variables & API Keys

Before running the application, make sure to set up the following:

* **Hugging Face API Key**:
  Obtain your Hugging Face API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set it as an environment variable:

  ```bash
  export HUGGINGFACE_API_KEY=your_huggingface_api_key_here
  ```

* **Google API Credentials (for Calendar Integration)**:
  Download your `calendar_service_account.json` file from the Google Cloud Console and place it in a `credentials/` folder:

  ```
  Agentic_chatbot/
  ├── credentials
      └── calendar_service_account.json
  ```

Alternatively, you can customize your `config.py` or `.env` file to load these values at runtime.

---

## 🗂️ Project Structure

```
Agentic_chatbot/
├── agents.py             # Defines agents for task handling
├── calendar_invite.py    # Handles calendar scheduling functionalities
├── config.py             # Configuration settings
├── environment.yml       # Conda environment specifications
├── model.py              # Core model implementations
├── rag_util.py           # Utilities for RAG operations
├── requirements.txt      # Python package dependencies
└── streamlit_app.py      # Streamlit web application
```

## 🛠️ Usage

Upon running the Streamlit application, you can interact with the chatbot through your web browser. The chatbot is capable of:

* Answering queries using retrieved contextual information.
* Scheduling events and managing calendar invites.
* Handling complex tasks through coordinated agent actions.

