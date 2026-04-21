# Creative Agent — Streamlit deployment

## Files
- `streamlit_app.py`
- `requirements.txt`

## Local run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to Streamlit Community Cloud.
3. Click **Create app**.
4. Select the repository, branch, and `streamlit_app.py`.
5. In **Secrets**, add:
   ```toml
   OPENAI_API_KEY = "your_key_here"
   OPENAI_MODEL = "gpt-5.4-mini"
   ```
6. Deploy.
