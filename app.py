import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from llm_client import call_llm, compare_models, AVAILABLE_MODELS
from analytics import (get_summary_stats, get_model_comparison,
                        get_calls_over_time, get_latency_over_time,
                        get_token_efficiency, get_cost_by_model,
                        get_call_history)
from evaluator import evaluate_comparison

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LLM Analytics Dashboard")
st.caption("Compare models · Track costs · Evaluate response quality")

# ── Top KPI cards ─────────────────────────────────────────────────────────────

stats = get_summary_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Calls",      stats["total_calls"])
c2.metric("Total Tokens",     f"{stats['total_tokens']:,}")
c3.metric("Total Cost (USD)", f"${stats['total_cost_usd']:.6f}")
c4.metric("Avg Latency",      f"{stats['avg_latency_ms']} ms")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔬 Compare Models", "📊 Analytics", "📋 Call History"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Compare Models
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Side-by-Side Model Comparison")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        prompt = st.text_area(
            "Enter your prompt",
            placeholder="e.g. Explain transformer architecture in simple terms.",
            height=120,
        )
    with col_right:
        selected_models = st.multiselect(
            "Models to compare",
            options=AVAILABLE_MODELS,
            default=AVAILABLE_MODELS,
        )
        session_label = st.text_input("Session label (optional)", value="comparison")
        run_btn = st.button("▶ Run Comparison", type="primary", use_container_width=True)

    if run_btn:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        elif not selected_models:
            st.warning("Select at least one model.")
        else:
            with st.spinner("Calling models..."):
                results = compare_models(prompt, models=selected_models,
                                         session_label=session_label)
                results = evaluate_comparison(results)

            st.success(f"Done! Called {len(results)} models.")

            # ── Score bar chart ───────────────────────────────────────────
            score_data = [{
                "Model":        r["model"],
                "Overall":      r["evaluation"]["overall"],
                "Relevance":    r["evaluation"]["relevance"],
                "Completeness": r["evaluation"]["completeness"],
                "Conciseness":  r["evaluation"]["conciseness"],
                "Readability":  r["evaluation"]["readability"],
                "Confidence":   r["evaluation"]["confidence"],
            } for r in results]

            score_df = pd.DataFrame(score_data)
            fig_scores = px.bar(
                score_df.melt(id_vars="Model", var_name="Dimension", value_name="Score"),
                x="Model", y="Score", color="Dimension", barmode="group",
                title="Quality Scores by Model & Dimension",
                height=350,
            )
            st.plotly_chart(fig_scores, use_container_width=True)

            # ── Per-model cards ───────────────────────────────────────────
            cols = st.columns(len(results))
            for col, r in zip(cols, results):
                with col:
                    overall = r["evaluation"]["overall"]
                    color   = "🟢" if overall >= 75 else "🟡" if overall >= 55 else "🔴"
                    st.markdown(f"**{r['model']}** {color} **{overall}/100**")
                    st.caption(
                        f"⏱ {r['latency_ms']} ms &nbsp;|&nbsp; "
                        f"🔤 {r['total_tokens']} tokens &nbsp;|&nbsp; "
                        f"💰 ${r['cost_usd']}"
                    )
                    st.text_area(
                        "Response",
                        value=r["response"],
                        height=200,
                        key=f"resp_{r['model']}",
                        disabled=True,
                    )
                    # Mini score breakdown
                    e = r["evaluation"]
                    mini_df = pd.DataFrame({
                        "Dimension": ["Relevance", "Completeness",
                                      "Conciseness", "Readability", "Confidence"],
                        "Score": [e["relevance"], e["completeness"],
                                  e["conciseness"], e["readability"], e["confidence"]],
                    })
                    fig_mini = px.bar(mini_df, x="Score", y="Dimension",
                                      orientation="h", height=200,
                                      color="Score", color_continuous_scale="RdYlGn",
                                      range_color=[0, 100])
                    fig_mini.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0),
                                           coloraxis_showscale=False)
                    st.plotly_chart(fig_mini, use_container_width=True)

    # ── Single model quick call ───────────────────────────────────────────────
    with st.expander("⚡ Quick Single Call"):
        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            q_prompt = st.text_input("Prompt", key="quick_prompt")
        with q_col2:
            q_model = st.selectbox("Model", AVAILABLE_MODELS, key="quick_model")
        if st.button("Run", key="quick_run"):
            if q_prompt.strip():
                with st.spinner("Calling..."):
                    r = call_llm(q_prompt, model=q_model)
                st.markdown(f"**Response** ({r['latency_ms']} ms · {r['total_tokens']} tokens)")
                st.write(r["response"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Usage Analytics")

    # Model comparison table
    model_df = get_model_comparison()
    if not model_df.empty:
        st.markdown("#### Model Performance Benchmarks")
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)

        # Latency bar chart
        with c1:
            fig_lat = px.bar(
                model_df, x="model", y="avg_latency_ms",
                title="Average Latency per Model (ms)",
                color="avg_latency_ms",
                color_continuous_scale="RdYlGn_r",
                labels={"avg_latency_ms": "Avg Latency (ms)", "model": "Model"},
            )
            fig_lat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_lat, use_container_width=True)

        # Token usage bar chart
        with c2:
            fig_tok = px.bar(
                model_df, x="model", y=["avg_tokens"],
                title="Average Token Usage per Model",
                labels={"value": "Tokens", "model": "Model"},
                barmode="group",
            )
            st.plotly_chart(fig_tok, use_container_width=True)

        # Token efficiency
        st.markdown("#### Token Efficiency (Response / Prompt ratio)")
        eff_df = get_token_efficiency()
        fig_eff = px.bar(
            eff_df, x="model", y="response_ratio",
            title="Response/Prompt Token Ratio — Higher = More Verbose",
            color="response_ratio", color_continuous_scale="Blues",
            labels={"response_ratio": "Ratio", "model": "Model"},
        )
        fig_eff.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_eff, use_container_width=True)

        # Cost breakdown
        st.markdown("#### Cost Breakdown")
        cost_df = get_cost_by_model()
        c3, c4 = st.columns(2)
        with c3:
            fig_cost = px.pie(
                cost_df[cost_df["total_cost_usd"] > 0],
                names="model", values="total_cost_usd",
                title="Cost Share by Model",
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        with c4:
            st.dataframe(cost_df, use_container_width=True, hide_index=True)

        # Latency over time
        st.markdown("#### Latency Trend Over Time")
        lat_time_df = get_latency_over_time()
        if not lat_time_df.empty:
            fig_trend = px.line(
                lat_time_df, x="timestamp", y="latency_ms",
                color="model", markers=True,
                title="Latency (ms) per Call",
                labels={"latency_ms": "Latency (ms)", "timestamp": "Time"},
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No data yet. Run some comparisons in the Compare tab first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Call History
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Call History Log")

    h_col1, h_col2 = st.columns([2, 1])
    with h_col1:
        model_filter = st.selectbox(
            "Filter by model",
            options=["All"] + AVAILABLE_MODELS,
            key="hist_filter",
        )
    with h_col2:
        limit = st.slider("Rows to show", 10, 200, 50, key="hist_limit")

    mf = None if model_filter == "All" else model_filter
    history_df = get_call_history(limit=limit, model_filter=mf)

    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv = history_df.to_csv(index=False)
        st.download_button(
            "⬇ Export as CSV",
            data=csv,
            file_name="llm_call_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No calls logged yet.")
