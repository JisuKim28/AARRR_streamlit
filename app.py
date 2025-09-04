# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from pathlib import Path
# import pandas as pd

# st.set_page_config(page_title="Google Merchandise Sale Prediction", layout="wide")

# # 앱 전체 메인 타이틀

# st.markdown(
#     """
#     <h1 style="font-size:40px; font-weight:bold;">
#         <span style="color:#4285F4;">G</span>
#         <span style="color:#DB4437;">o</span>
#         <span style="color:#F4B400;">o</span>
#         <span style="color:#4285F4;">g</span>
#         <span style="color:#0F9D58;">l</span>
#         <span style="color:#DB4437;">e</span>
#         Merchandise Sale Prediction
#     </h1>
#     """,
#     unsafe_allow_html=True
# )

# # 구분선
# st.markdown(
#     "<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>",
#     unsafe_allow_html=True
# )


# st.set_page_config(page_title="Overview", layout="wide")
# st.title("Overview")

# # CSV 파일 직접 읽기
# CSV_PATH = r"C:\Users\jisus\OneDrive\바탕 화면\Python\google_with_cluster3_aha.csv"
# df = pd.read_csv(CSV_PATH)

# # ---Overview---

# # 세션 ID 생성 (user + visitStartTime 조합)
# df["sessionId"] = df["fullVisitorId"].astype(str) + "_" + df["visitStartTime"].astype(str)

# # --- KPI 계산 ---
# total_sessions = df["sessionId"].nunique()
# unique_users = df["fullVisitorId"].nunique()

# # 세션/유저 비율
# sessions_per_user = total_sessions / unique_users if unique_users else 0

# # --- KPI 출력 ---
# c1, c2, c3 = st.columns(3)
# c1.metric("총 세션 수", f"{total_sessions:,}")
# c2.metric("총 유저 수", f"{unique_users:,}")
# c3.metric("유저당 평균 방문 횟수", f"{sessions_per_user:.2f}")

# # --- 평균 체류 시간 / 페이지뷰 시간 ---
# st.set_page_config(page_title="Overview", layout="wide")

# # 세션 ID 생성 (user + visitStartTime 조합)
# df["sessionId"] = df["fullVisitorId"].astype(str) + "_" + df["visitStartTime"].astype(str)

# # --- KPI 계산 ---
# # 평균 체류시간 (분 단위)
# avg_time_minutes = df["totalTimeOnSite"].mean() / 60

# # 세션당 평균 페이지뷰
# avg_pageviews = df["totalPageviews"].mean()

# # --- KPI 출력 ---
# c1, c2 = st.columns(2)
# c1.metric("평균 체류시간 (분)", f"{avg_time_minutes:.1f}")
# c2.metric("세션당 평균 페이지뷰", f"{avg_pageviews:.2f}")

# # ----- 신규/재방문 비율, 카트전환율 ---

# # 방문 유형 라벨링
# df["visit_type"] = np.where(pd.to_numeric(df["isFirstVisit"], errors="coerce") == 1, "신규", "재방문")

# # addedToCart 불리언 변환
# added_num = pd.to_numeric(df["addedToCart"], errors="coerce")
# added_str = df["addedToCart"].astype(str).str.lower()
# df["cart_flag"] = (added_num.fillna(0) > 0) | (added_str.isin(["1", "true", "t", "y", "yes"]))

# # ✅ summary 생성
# summary = (
#     df.groupby("visit_type")
#       .agg(
#           sessions=("fullVisitorId", "count"),
#           cart_sessions=("cart_flag", "sum")
#       )
#       .reset_index()
# )
# summary["cart_rate"] = summary["cart_sessions"] / summary["sessions"]

# # 보기 좋게 포맷
# summary_view = summary.copy()
# summary_view["session_ratio(%)"] = (
#     summary_view["sessions"] / summary_view["sessions"].sum() * 100
# ).round(1).astype(str) + "%"
# summary_view["cart_rate(%)"] = (summary_view["cart_rate"] * 100).round(1).astype(str) + "%"

# st.subheader("신규/재방문 회원별 Cart 전환율")

# # 좌우 배치
# left, right = st.columns([1, 1])

# with left:
#     st.dataframe(
#         summary_view[["visit_type", "session_ratio(%)", "cart_rate(%)"]],
#         use_container_width=True
#     )

# with right:
#     fig = px.bar(
#         summary,
#         x="visit_type",
#         y="cart_rate",
#         text=(summary["cart_rate"] * 100).round(1).astype(str) + "%",
#         labels={"visit_type": "방문 유형", "cart_rate": "Cart 전환율"},
#         category_orders={"visit_type": ["신규", "재방문"]},
#         color="visit_type",  # 색 기준 컬럼
#         color_discrete_map={
#             "신규": "#4285F4",     # 구글 파란색
#             "재방문": "#F4B400"   # 구글 노란색
#         }
#     )
#     fig.update_yaxes(tickformat=".0%")
#     fig.update_traces(textposition="outside")
#     fig.update_layout(
#         bargap=0.7,   # 막대 간격
#         bargroupgap=0.1
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # --- AHA 달성률 ---
# st.subheader("전체 AHA 달성률")

# # AhaMoment를 견고하게 불리언으로 변환
# s = df["AhaMoment"]
# if pd.api.types.is_bool_dtype(s):
#     aha = s.fillna(False)
# else:
#     num = pd.to_numeric(s, errors="coerce").fillna(0)
#     txt = s.astype(str).str.strip().str.lower()
#     aha = (num > 0) | (txt.isin(["true", "t", "y", "yes", "1"]))

# aha_rate = aha.mean()  # TRUE 비율

# # KPI
# st.metric("Aha 달성률", f"{aha_rate*100:.1f}%")

# # 파이차트 (TRUE vs FALSE)
# pie_df = pd.DataFrame({
#     "Aha": ["TRUE", "FALSE"],
#     "count": [int(aha.sum()), int((~aha).sum())]
# })

# fig = px.pie(
#     pie_df,
#     names="Aha",
#     values="count",
#     hole=0.35,
#     color="Aha",
#     color_discrete_map={
#         "TRUE": "#F4B400",   # 노랑
#         "FALSE": "#4285F4"   # 파랑
#     }
# )

# fig.update_traces(
#     textposition="inside",
#     texttemplate="%{label}: %{percent:.1%}"
# )

# # 👉 차트와 설명을 나란히 배치
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.markdown(
#         """
#         **AHA 모먼트 측정 기준**  
#         - 처음으로 상품을 장바구니에 담았을 때  
#         - 세션 시간(TimePerSession)이 일정 기준(예: 3 이상)에 도달했을 때
#         """
#     )


# # --- 퍼널 분석---
# import streamlit as st
# import pandas as pd
# import plotly.express as px

# st.subheader("퍼널 분석")

# step1 = (df["totalPageviews"] >= 1)
# step2 = (df["productPagesViewed"] >= 1)
# step3 = (pd.to_numeric(df["addedToCart"], errors="coerce").fillna(0) >= 1)

# n1 = int(step1.sum())
# n2 = int((step1 & step2).sum())
# n3 = int((step1 & step2 & step3).sum())

# conv12 = n2 / n1 if n1 else 0
# conv23 = n3 / n2 if n2 else 0
# conv_overall = n3 / n1 if n1 else 0

# k1, k2, k3 = st.columns(3)
# k1.metric("페이지뷰→제품상세 전환율", f"{conv12*100:.1f}%")
# k2.metric("제품상세→카트추가 전환율", f"{conv23*100:.1f}%")
# k3.metric("페이지뷰→카트추가 전환율", f"{conv_overall*100:.1f}%")

# # 퍼널 데이터프레임 (그대로)
# funnel_df = pd.DataFrame({
#     "Step": ["페이지뷰 ≥1", "제품상세 ≥1", "카트추가 ≥1"],
#     "Sessions": [n1, n2, n3]
# })

# # 이전 단계 대비 %
# pct_prev = [1.0, (n2 / n1) if n1 else 0, (n3 / n2) if n2 else 0]
# # 천단위 콤마 + 퍼센트
# funnel_df["Text"] = [f"{v:,.0f} ({p*100:.1f}%)" for v, p in zip(funnel_df["Sessions"], pct_prev)]

# # ✅ 기본 value 라벨(k 단위)을 없애고, 우리가 만든 텍스트만 표시
# import plotly.express as px
# fig_funnel = px.funnel(funnel_df, x="Sessions", y="Step")

# fig_funnel.update_traces(
#     text=funnel_df["Text"],       # 표기할 텍스트 직접 지정
#     texttemplate="%{text}",       # 우리가 만든 텍스트만 표시
#     textposition="inside",
#     textinfo="none",              # 기본 textinfo 제거
#     marker_color="#4285F4"        # 👉 퍼널 색상 파란색으로 통일
# )

# st.plotly_chart(fig_funnel, use_container_width=True)

# st.markdown(
#     "<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>",
#     unsafe_allow_html=True
# )    

# # --- Cluster Summary Table ---
# # 퍼널 차트
# st.set_page_config(page_title="Cluster Summary", layout="wide")
# st.title("Cluster Summary Table")

# # 표 데이터 정의 + 클러스터링 결과 추가
# data = {
#     "고객군": [
#         "탐색형 고객 (Explorers)", 
#         "비활성 고객 (Visitors)", 
#         "충성/핵심 고객 (Core Buyers)"
#     ],
#     "유저 수": ["260,180", "305,542", "29,126"],
#     "Recency (일)": [159.0, 164.4, 150.4],
#     "Frequency (방문일수)": [1.21, 1.07, 1.75],
#     "Cart Rate": [0.01, 0.00, 0.77],
#     "Search Rate": [0.96, 0.01, 0.95],
#     "Time/Session (정규화)": [1.19, 0.01, 5.32]
# }

# df_cluster = pd.DataFrame(data)

# # Streamlit에 표시
# st.subheader("클러스터별 요약")
# st.dataframe(df_cluster, use_container_width=True)

# # --- 클러스터별 유저 비중 ---
# st.subheader("클러스터별 유저 비중")

# # 데이터 정의
# data = {
#     "Cluster": [0, 1, 2],
#     "클러스터링 결과": [
#         "탐색형 고객 (Explorers)",
#         "비활성 고객 (Visitors)",
#         "충성 고객 (Core Buyers)"
#     ],
#     "유저 수": [260180, 305542, 29126]  # 숫자로 변환
# }

# df_cluster = pd.DataFrame(data)

# # Pie 차트
# # 라벨 정규화 (앞뒤 공백 제거)
# df_cluster["label"] = df_cluster["클러스터링 결과"].str.strip()

# # 원하는 순서 (탐색=파랑, 비활성=노랑, 충성=빨강)
# order = ["탐색형 고객 (Explorers)", "비활성 고객 (Visitors)", "충성/핵심 고객 (Core Buyers)"]
# colors = ["#4285F4", "#F4B400", "#DB4437"]   # Google Blue, Yellow, Red

# fig = px.pie(
#     df_cluster,
#     names="label",
#     values="유저 수",
#     hole=0.3,
#     category_orders={"label": order},                 # 라벨 순서 고정
#     color="label",                                    # 색 기준 컬럼
#     color_discrete_sequence=colors                    # 순서에 맞춘 색 적용
# )

# fig.update_traces(
#     textinfo="label+percent",
#     pull=[0.02, 0.02, 0.05],
#     rotation=90
# )

# fig.update_layout(
#     margin=dict(t=40, b=40, l=40, r=40),
#     legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
# )

# st.plotly_chart(fig, use_container_width=True)

# # 구분선
# st.markdown(
#     "<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>",
#     unsafe_allow_html=True
# )

# # app.py — AARRR Dashboard (USER-level dedup with cluster population by unique users)
# # 요구 컬럼 예: fullVisitorId, date, cluster, productPagesViewed, addedToCart,
# #               trafficMedium/trafficMed/trafficSource, deviceCategory, country,
# #               AhaMoment, (선택) TimePerSession_norm/TimePerSessionNorm

# import os, uuid
# from typing import Optional
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# from pandas.api.types import is_datetime64tz_dtype

# # ---------------- Page/UI ----------------
# st.set_page_config(page_title="AARRR Dashboard", layout="wide")
# st.title("AARRR Dashboard — USER level")

# # -------- Colors & Cluster map --------
# GOOGLE_BLUE   = "#4285F4"    # single blue for funnel & bars
# GOOGLE_YELLOW = "#F4B400"    # cluster 1
# GOOGLE_RED    = "#DB4437"    # cluster 2
# GOOGLE_GREY   = "#9AA0A6"
# CLUSTER_COLORS = {"0": GOOGLE_BLUE, "1": GOOGLE_YELLOW, "2": GOOGLE_RED}
# DESIRED_ORDER = ["0", "1", "2"]

# def ukey(name): return f"{name}-{uuid.uuid4().hex[:8]}"

# # ---------------- Loader helpers ----------------
# NEEDED_BASE_COLS = {
#     "fullVisitorId","date","cluster",
#     "productPagesViewed","addedToCart",
#     "deviceCategory","country",
#     "trafficMedium","trafficMed","trafficSource",
#     "AhaMoment",
#     "TimePerSessionNorm","TimePerSession_norm","time_per_session",
#     "totalTimeOnSite","totalVisits",
#     # 👇 도시 컬럼 후보 추가 (없으면 로더에서 잘려 나갑니다!)
#     "city", "City", "regionCity",
# }


# def _seek0(x):
#     try: x.seek(0)
#     except Exception: pass

# @st.cache_data(show_spinner=False, max_entries=1)
# def load_csv_smart(source, is_path: bool):
#     if not is_path: _seek0(source)
#     header = pd.read_csv(source, nrows=0, engine="python", on_bad_lines="skip")
#     cols = [c.strip() for c in header.columns]
#     usecols = [c for c in cols if c in NEEDED_BASE_COLS]

#     must = {"fullVisitorId","date","cluster","productPagesViewed","addedToCart","AhaMoment"}
#     missing = [c for c in must if c not in cols]
#     if missing:
#         raise RuntimeError(f"필수 컬럼 누락: {missing}")

#     dtype_map = {"fullVisitorId":"string", "date":"string"}
#     try:
#         if not is_path: _seek0(source)
#         return pd.read_csv(source, engine="pyarrow", usecols=usecols, dtype=dtype_map)
#     except Exception:
#         pass
#     try:
#         if not is_path: _seek0(source)
#         return pd.read_csv(source, engine="python", usecols=usecols,
#                            dtype=dtype_map, on_bad_lines="skip")
#     except Exception:
#         pass

#     if not is_path: _seek0(source)
#     chunks = []
#     for chunk in pd.read_csv(source, engine="python", usecols=usecols,
#                              dtype=dtype_map, on_bad_lines="skip", chunksize=50_000):
#         chunks.append(chunk)
#     return pd.concat(chunks, ignore_index=True)

# def to_bool(s: pd.Series) -> pd.Series:
#     if s.dtype == bool: return s
#     try: return (pd.to_numeric(s, errors="coerce").fillna(0) > 0)
#     except Exception:
#         ss = s.astype(str).str.lower().str.strip()
#         return ss.isin({"1","true","t","y","yes"})

# def ensure_dt(s: pd.Series) -> pd.Series:
#     out = pd.to_datetime(s, errors="coerce")
#     if is_datetime64tz_dtype(out):
#         try: out = out.dt.tz_convert("UTC").dt.tz_localize(None)
#         except AttributeError: out = out.dt.tz_localize(None)
#     return out

# # ---------------- Data load ----------------
# st.sidebar.header("데이터")
# default_path = "./google_with_cluster3_aha.csv"
# up   = st.sidebar.file_uploader("CSV 업로드(선택)", type=["csv"])
# path = st.sidebar.text_input("경로(선택)", value=default_path)

# if up is not None:
#     df = load_csv_smart(up, is_path=False)
#     st.success("업로드한 파일을 불러왔습니다.")
# else:
#     if not os.path.exists(path):
#         st.error(f"경로가 존재하지 않습니다: {path}"); st.stop()
#     df = load_csv_smart(path, is_path=True)
#     st.info(f"기본 경로에서 로드: {path}")

# if df.empty:
#     st.error("빈 데이터입니다."); st.stop()

# # ---------------- Preprocess ----------------
# user_col     = "fullVisitorId"
# date_col     = "date"
# cluster_col  = "cluster"
# detail_col   = "productPagesViewed"
# cart_col     = "addedToCart"
# device_col   = "deviceCategory"
# country_col  = "country"

# # channel column
# if "trafficMedium" in df.columns:
#     medium_col = "trafficMedium"
# elif "trafficMed" in df.columns:
#     medium_col = "trafficMed"
# elif "trafficSource" in df.columns:
#     medium_col = "trafficSource"
# else:
#     st.error("채널 차원을 위한 trafficMedium/trafficMed/trafficSource 중 하나가 필요합니다."); st.stop()

# # session time column (optional)
# sess_time_col = "TimePerSession_norm" if "TimePerSession_norm" in df.columns \
#     else ("TimePerSessionNorm" if "TimePerSessionNorm" in df.columns else None)

# # type cleaning
# session_col = "_session_surrogate"
# df = df.copy()
# df[user_col] = df[user_col].astype("string").fillna("")
# cnum = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")
# df = df[cnum.isin([0,1,2])].copy()
# df[cluster_col] = cnum.astype("Int64").astype(str)

# df[date_col] = ensure_dt(df[date_col])
# df = df[df[date_col].notna()]
# df[detail_col] = pd.to_numeric(df[detail_col], errors="coerce").fillna(0)
# df[cart_col]   = to_bool(df[cart_col])
# df["AhaMoment"] = to_bool(df["AhaMoment"])

# # session surrogate (30m bucket if time present, else date)
# if df[date_col].dt.time.astype(str).nunique() > 1:
#     bucket = df[date_col].dt.floor("30min").astype(str)
# else:
#     bucket = df[date_col].dt.strftime("%Y-%m-%d")
# df[session_col] = (df[user_col].astype(str) + "|" + bucket)

# # ---------------- Date filter ----------------
# st.sidebar.header("기간")
# min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
# start = st.sidebar.date_input("시작일", value=min_d)
# end   = st.sidebar.date_input("종료일", value=max_d)

# start_dt = pd.Timestamp(start)
# end_dt   = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
# dfp = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
# if dfp.empty:
#     st.warning("선택 구간 데이터 없음"); st.stop()

# # ---------------- Representative cluster (mode) for user snapshot ----------------
# def rep_mode(s: pd.Series) -> Optional[str]:
#     m = pd.to_numeric(s, errors="coerce"); m = m[m.notna()]
#     if m.empty: return None
#     return str(int(m.mode().iat[0]))

# # 최신 스냅샷(유저 최신 행) — Activation/분모 용
# df_user_last = (
#     dfp.sort_values([user_col, date_col])
#        .drop_duplicates(subset=[user_col], keep="last")
#        [[user_col, cluster_col, "AhaMoment", cart_col]]
#        .copy()
# )
# df_user_last[cluster_col] = df_user_last[cluster_col].astype(str)
# df_user_last["AhaMoment"] = to_bool(df_user_last["AhaMoment"])
# df_user_last["cart_bool"] = to_bool(df_user_last[cart_col])

# # 기간 전체 행동(any) + 대표 클러스터(최빈값) — 퍼널/리텐션 용
# rep_cluster_mode = (dfp.groupby(user_col)[cluster_col].apply(rep_mode)
#                        .reset_index(name=cluster_col))
# user_any = (dfp.groupby(user_col)
#               .agg(detail_any=(detail_col, lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).any())),
#                    cart_any=(cart_col, "any"))
#               .reset_index())
# uf = user_any.merge(rep_cluster_mode, on=user_col, how="left")
# uf["home"] = 1

# # # =========================================================
# # # 1) Overview
# # # =========================================================
# # st.header("1. Overview (USER 기준)")

# # # session-level funnel (reference)
# # sess_agg = dfp.groupby(session_col).agg(
# #     detail_any=(detail_col, lambda s: (s>0).any()),
# #     cart_any=(cart_col, "any"),
# # )
# # home_cnt = int(len(sess_agg))
# # detail_cnt = int(sess_agg["detail_any"].sum())
# # cart_cnt   = int(sess_agg["cart_any"].sum())

# # m1,m2,m3,m4 = st.columns(4)
# # m1.metric("총 세션", f"{home_cnt:,}")
# # m2.metric("총 유저", f"{dfp[user_col].nunique():,}")
# # # Aha 경험률(유저, 최신 스냅샷)
# # aha_overall_rate = (df_user_last["AhaMoment"].mean() if len(df_user_last) else np.nan)
# # m3.metric("Aha 경험률(유저, 최신 스냅샷)", f"{aha_overall_rate:.1%}" if pd.notna(aha_overall_rate) else "NA")
# # m4.metric("세션 퍼널 전환율", f"{(cart_cnt/home_cnt):.1%}" if home_cnt else "NA")

# # st.plotly_chart(go.Figure(go.Funnel(
# #     y=["Home(세션)","Detail","Cart"],
# #     x=[home_cnt, detail_cnt, cart_cnt],
# #     textposition="inside", textinfo="value+percent previous"
# # )), use_container_width=True, key=ukey("overview-funnel"))

# # if sess_time_col is not None:
# #     st.plotly_chart(px.histogram(
# #         dfp.groupby(session_col)[sess_time_col].mean().reset_index(),
# #         x=sess_time_col, nbins=40, title="세션 평균 체류시간(정규화) 분포",
# #         color_discrete_sequence=[GOOGLE_BLUE]
# #     ), use_container_width=True, key=ukey("overview-time"))

# # with st.expander("원본 집계 점검 (행/유저/세션 surrogate)"):
# #     st.write({
# #         "rows_after_filter": len(dfp),
# #         "unique_users": int(dfp[user_col].nunique()),
# #         "sessions_surrogate": int(dfp[session_col].nunique())
# #     })

# # =========================================================
# # 2) Activation — 분모=클러스터 인원(유저 스냅샷)
# # =========================================================
# st.header("1. Activation (분모=클러스터 인원, 유저 최신 스냅샷)")

# # 분모(클러스터 인원)
# cluster_pop = (df_user_last.groupby(cluster_col)[user_col]
#                .nunique().rename("cluster_users").reset_index())
# cluster_pop[cluster_col] = pd.Categorical(cluster_pop[cluster_col], categories=DESIRED_ORDER, ordered=True)
# cluster_pop = cluster_pop.sort_values(cluster_col)

# # Aha true
# aha_true = (df_user_last.groupby(cluster_col)["AhaMoment"]
#             .sum().rename("aha_true").reset_index())

# aha_rate_den_cluster = (cluster_pop.merge(aha_true, on=cluster_col, how="left")
#                         .fillna({"aha_true":0})
#                         .assign(aha_rate=lambda t: t["aha_true"] / t["cluster_users"].replace(0, np.nan))
#                         [[cluster_col,"cluster_users","aha_true","aha_rate"]])
# aha_rate_den_cluster[cluster_col] = pd.Categorical(
#     aha_rate_den_cluster[cluster_col], categories=DESIRED_ORDER, ordered=True)
# aha_rate_den_cluster = aha_rate_den_cluster.sort_values(cluster_col)

# # Aha→Cart (전체 대비)
# aha_cart_true = ((df_user_last["AhaMoment"] & df_user_last["cart_bool"])
#                  .groupby(df_user_last[cluster_col]).sum()
#                  .rename("aha_cart_true").reset_index())

# aha_cart_overall_den_cluster = (cluster_pop.merge(aha_cart_true, on=cluster_col, how="left")
#                                 .fillna({"aha_cart_true":0})
#                                 .assign(aha_cart_overall=lambda t: t["aha_cart_true"] / t["cluster_users"].replace(0, np.nan))
#                                 [[cluster_col,"cluster_users","aha_cart_true","aha_cart_overall"]])
# aha_cart_overall_den_cluster[cluster_col] = pd.Categorical(
#     aha_cart_overall_den_cluster[cluster_col], categories=DESIRED_ORDER, ordered=True)
# aha_cart_overall_den_cluster = aha_cart_overall_den_cluster.sort_values(cluster_col)

# # Aha 달성률 차트
# fig_aha = px.bar(
#     aha_rate_den_cluster, y=cluster_col, x="aha_rate", orientation="h",
#     title="클러스터별 Aha 달성률 (분모=클러스터 인원)",
#     color=cluster_col, color_discrete_map=CLUSTER_COLORS,
#     labels={"aha_rate":"Aha 달성률", cluster_col:"Cluster"}
# )
# fig_aha.update_yaxes(categoryorder="array", categoryarray=DESIRED_ORDER)
# fig_aha.update_traces(texttemplate="%{x:.1%}", textposition="outside", showlegend=False)
# fig_aha.update_layout(xaxis_tickformat=".1%")
# st.plotly_chart(fig_aha, use_container_width=True, key=ukey("act-aha"))

# # Aha→Cart (전체 대비) 차트
# fig_aha_cart = px.bar(
#     aha_cart_overall_den_cluster, y=cluster_col, x="aha_cart_overall", orientation="h",
#     title="클러스터별 Aha → Cart (분모=클러스터 인원)",
#     color=cluster_col, color_discrete_map=CLUSTER_COLORS,
#     labels={"aha_cart_overall":"Aha→Cart (전체 대비)", cluster_col:"Cluster"}
# )
# fig_aha_cart.update_yaxes(categoryorder="array", categoryarray=DESIRED_ORDER)
# fig_aha_cart.update_traces(texttemplate="%{x:.1%}", textposition="outside", showlegend=False)
# fig_aha_cart.update_layout(xaxis_tickformat=".1%")
# st.plotly_chart(fig_aha_cart, use_container_width=True, key=ukey("act-aha-cart"))

# # 숫자 표
# st.dataframe(
#     aha_rate_den_cluster.merge(
#         aha_cart_overall_den_cluster[[cluster_col,"aha_cart_true","aha_cart_overall"]],
#         on=cluster_col, how="left"
#     ).rename(columns={
#         "cluster_users":"클러스터 인원",
#         "aha_true":"Aha True 수",
#         "aha_rate":"Aha 달성률",
#         "aha_cart_true":"Aha∩Cart True 수",
#         "aha_cart_overall":"Aha→Cart (전체 대비)"
#     })[[cluster_col,"클러스터 인원","Aha True 수","Aha 달성률","Aha∩Cart True 수","Aha→Cart (전체 대비)"]],
#     use_container_width=True
# )

# # =========================================================
# # 3) Funnel (USER 기준 / 홈 기준, 게이트식) — 클러스터별만 표시
# # =========================================================
# st.subheader("퍼널 전환율")

# # 전체 탭 제거 → Cluster 0, 1, 2만
# tabs = st.tabs(["Cluster 0", "Cluster 1", "Cluster 2"])
# tab_keys = ["0", "1", "2"]

# def draw_home_based_funnel(uv: pd.DataFrame, title: str, key_sfx: str):
#     h = int(len(uv))
#     d = int((uv["detail_any"] == 1).sum())
#     k = int(((uv["detail_any"] == 1) & (uv["cart_any"] == 1)).sum())

#     pct_detail = (d / h) if h else np.nan
#     pct_cart   = (k / h) if h else np.nan

#     seg_text = [
#         f"{h:,}\n100%",
#         f"{d:,}\n{pct_detail:.1%} (Home→Detail)",
#         f"{k:,}\n{pct_cart:.1%} (Home→Cart via Detail)",
#     ]

#     fig = go.Figure(go.Funnel(
#         y=["Home","Detail","Cart(Detail 경유)"],
#         x=[h, d, k],
#         text=seg_text,
#         textposition="inside",
#         textinfo="text",  # Plotly 기본 percent 제거
#         marker={"color": [GOOGLE_BLUE, GOOGLE_BLUE, GOOGLE_BLUE]},
#         hovertemplate="%{y}: %{x:,}<extra></extra>",
#     ))
#     fig.update_layout(title=title)
#     st.plotly_chart(fig, use_container_width=True, key=ukey(f"funnel-{key_sfx}"))

#     st.dataframe(pd.DataFrame([{
#         "home_users": h,
#         "detail_users": d,
#         "cart_users": k,
#         "Home→Detail(%)": round((pct_detail * 100), 1) if h else np.nan,
#         "Home→Cart(Detail 경유)(%)": round((pct_cart * 100), 1) if h else np.nan,
#         "Detail→Cart(%)": round(((k / d) * 100), 1) if d else np.nan,
#     }]), use_container_width=True)

# # 클러스터별 탭만
# for t, key in zip(tabs, tab_keys):
#     with t:
#         uv = uf[uf[cluster_col] == key].copy()
#         draw_home_based_funnel(uv, f"Funnel (Cluster {key})", key)

# # =========================================================
# # 4) Acquisition (USER 기준) — 탭 클릭 방식 (채널/디바이스/국가/도시)
# # =========================================================
# st.header("2. Acquisition (USER 기준)")

# # 공통 옵션
# cA, cB = st.columns([1,1])
# with cA:
#     min_share = st.slider("최소 유입 비중 제외", 0.0, 0.2, 0.01, 0.005, key=ukey("acq-minshare"))
# with cB:
#     top_n = st.number_input("TOP N", min_value=3, max_value=30, value=10, step=1, key=ukey("acq-topn"))

# # ----- 클린 유틸: 일반 차원(디바이스/국가 등)
# def _clean_series_exclude_noise(s: pd.Series) -> pd.Series:
#     ss = s.astype(str).str.strip()
#     bad = {"", "nan", "(none)", "none", "(not set)", "not set", "unavailable"}
#     return ss.where(~ss.str.lower().isin(bad), other=np.nan)

# # ----- 클린 유틸: 채널은 (none)을 유지 (빈값/nan/(not set)/unavailable만 제거)
# def _clean_channel_keep_none(s: pd.Series) -> pd.Series:
#     ss = s.astype(str).str.strip()
#     bad = {"", "nan", "(not set)", "not set", "unavailable"}
#     return ss.where(~ss.str.lower().isin(bad), other=np.nan)

# # ----- 공통 렌더 함수(비-채널용): 디바이스/국가 등은 기존 규칙 유지
# def render_acquisition_for_dim(dfp_src: pd.DataFrame, dim_col: str, title_prefix: str):
#     dfp_acq = dfp_src.copy()

#     # 유저 기준 유입 비중: 유저의 '최근' 해당 차원을 대표값으로 사용
#     rep_dim = (
#         dfp_acq[[user_col, dim_col, date_col]]
#         .sort_values([user_col, date_col])
#         .drop_duplicates(user_col, keep="last")[[user_col, dim_col]]
#         .copy()
#     )
#     rep_dim[dim_col] = _clean_series_exclude_noise(rep_dim[dim_col])

#     acq = (
#         rep_dim.dropna(subset=[dim_col])[dim_col]
#         .value_counts(normalize=True)
#         .rename_axis(dim_col)
#         .reset_index(name="share")
#         .sort_values("share", ascending=False)
#     )
#     acq = acq[acq["share"] >= float(min_share)].head(int(top_n))
#     order = acq[dim_col].tolist()

#     fig_share = px.bar(
#         acq, x=dim_col, y="share",
#         title=f"{title_prefix} 상위 {len(acq)} 유입 비중 (USER)",
#         color_discrete_sequence=[GOOGLE_BLUE]
#     )
#     fig_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
#     fig_share.update_layout(yaxis_tickformat=".1%")
#     fig_share.update_xaxes(categoryorder="array", categoryarray=order)
#     st.plotly_chart(fig_share, use_container_width=True, key=ukey(f"acq-share-{dim_col}"))

#     # 동일 카테고리 전환율 (세션→Cart)
#     sess_cart = (
#         dfp_acq.dropna(subset=[dim_col])
#               .groupby([dim_col, session_col])[cart_col].any()
#               .reset_index()
#     )
#     conv = (
#         sess_cart.groupby(dim_col)[cart_col].mean()
#                  .reset_index(name="conversion")
#     )
#     conv = conv[conv[dim_col].isin(order)]
#     conv[dim_col] = pd.Categorical(conv[dim_col], categories=order, ordered=True)
#     conv = conv.sort_values(dim_col)

#     fig_conv = px.bar(
#         conv, x=dim_col, y="conversion",
#         title=f"{title_prefix}별 전환율 (세션→장바구니, %)",
#         color_discrete_sequence=[GOOGLE_BLUE]
#     )
#     fig_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
#     fig_conv.update_layout(yaxis_tickformat=".1%")
#     fig_conv.update_xaxes(categoryorder="array", categoryarray=order)
#     st.plotly_chart(fig_conv, use_container_width=True, key=ukey(f"acq-conv-{dim_col}"))

# # 탭 구성: 채널 / 디바이스 / 국가 / 도시
# tab_channel, tab_device, tab_country, tab_city = st.tabs(["채널", "디바이스", "국가", "도시"])

# # ---------- 채널 탭: (none) 포함하여 유입 비중/전환율 렌더 ----------
# with tab_channel:
#     st.subheader("채널 (trafficMedium) — (none 포함)")

#     # USER 최신 스냅샷 기준 대표 채널
#     rep_medium = (
#         dfp[[user_col, medium_col, date_col]]
#         .sort_values([user_col, date_col])
#         .drop_duplicates(user_col, keep="last")[[user_col, medium_col]]
#         .copy()
#     )
#     rep_medium[medium_col] = _clean_channel_keep_none(rep_medium[medium_col])

#     channel_share = (
#         rep_medium.dropna(subset=[medium_col])[medium_col]
#         .value_counts(normalize=True)
#         .rename_axis(medium_col)
#         .reset_index(name="share")
#         .sort_values("share", ascending=False)
#     )
#     channel_share = channel_share[channel_share["share"] >= float(min_share)]
#     channel_share = channel_share.head(int(top_n))
#     order_channels = channel_share[medium_col].tolist()

#     fig_ch_share = px.bar(
#         channel_share, x=medium_col, y="share",
#         title=f"채널별 유입 비중 (USER) — Top {len(channel_share)}",
#         color_discrete_sequence=[GOOGLE_BLUE]
#     )
#     fig_ch_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
#     fig_ch_share.update_layout(yaxis_tickformat=".1%")
#     fig_ch_share.update_xaxes(categoryorder="array", categoryarray=order_channels)
#     st.plotly_chart(fig_ch_share, use_container_width=True, key=ukey("acq-channel-share"))

#     # 전환율(세션→Cart): (none) 포함해서 계산
#     df_conv = dfp.copy()
#     df_conv[medium_col] = _clean_channel_keep_none(df_conv[medium_col])
#     df_conv = df_conv.dropna(subset=[medium_col])

#     sess_cart = (
#         df_conv.groupby([medium_col, session_col])[cart_col].any()
#                .reset_index()
#     )
#     conv = (
#         sess_cart.groupby(medium_col)[cart_col]
#                  .mean().reset_index(name="conversion")
#     )
#     # 유입 Top 채널만 표시 + 같은 순서
#     conv = conv[conv[medium_col].isin(order_channels)]
#     conv[medium_col] = pd.Categorical(conv[medium_col], categories=order_channels, ordered=True)
#     conv = conv.sort_values(medium_col)

#     fig_ch_conv = px.bar(
#         conv, x=medium_col, y="conversion",
#         title="채널별 전환율 (세션→장바구니, %)",
#         color_discrete_sequence=[GOOGLE_BLUE]
#     )
#     fig_ch_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
#     fig_ch_conv.update_layout(yaxis_tickformat=".1%")
#     fig_ch_conv.update_xaxes(categoryorder="array", categoryarray=order_channels)
#     st.plotly_chart(fig_ch_conv, use_container_width=True, key=ukey("acq-channel-conv"))

# with tab_device:
#     render_acquisition_for_dim(dfp, device_col, "디바이스")

# with tab_country:
#     render_acquisition_for_dim(dfp, country_col, "국가")


# # --------------------------- 도시 탭 ---------------------------
# with tab_city:
#     # (none / not set / unavailable / demo placeholder) 값들 제거
#     def _clean_city(s: pd.Series) -> pd.Series:
#         ss = s.astype(str).str.strip()
#         bad = {
#             "", "nan",
#             "(none)", "none",
#             "(not set)", "not set",
#             "unavailable",
#             "not available in demo dataset",  # 👈 추가로 제외
#         }
#         return ss.where(~ss.str.lower().isin(bad), other=np.nan)

#     # dfp에 실제 존재하는 도시 컬럼 자동 선택
#     possible_city_cols = ["city", "City", "regionCity"]
#     city_col = next((c for c in possible_city_cols if c in dfp.columns), None)

#     # ---- 도시별 유입 TopN & 전환율 (상단) ----
#     if city_col is not None:
#         # 유입 비중 TopN (USER 최신 스냅샷)
#         rep_city = (
#             dfp[[user_col, city_col, date_col]]
#               .sort_values([user_col, date_col])
#               .drop_duplicates(user_col, keep="last")[[user_col, city_col]]
#               .copy()
#         )
#         rep_city[city_col] = _clean_city(rep_city[city_col])

#         top_n_cities = int(top_n) if "top_n" in globals() else 10
#         city_share = (
#             rep_city.dropna(subset=[city_col])[city_col]
#                     .value_counts(normalize=True)
#                     .rename_axis(city_col).reset_index(name="share")
#                     .sort_values("share", ascending=False)
#                     .head(top_n_cities)
#         )

#         if not city_share.empty:
#             order_cities = city_share[city_col].tolist()

#             fig_city_share = px.bar(
#                 city_share, x=city_col, y="share",
#                 title="도시별 유입 비중 Top N (USER)",
#                 color_discrete_sequence=[GOOGLE_BLUE]
#             )
#             fig_city_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
#             fig_city_share.update_layout(yaxis_tickformat=".1%")
#             fig_city_share.update_xaxes(categoryorder="array", categoryarray=order_cities)
#             st.plotly_chart(fig_city_share, use_container_width=True, key=ukey("city-share"))

#             # 도시별 전환율(세션→Cart) — 유입 상위 도시에 한정
#             df_city_conv = dfp.copy()
#             df_city_conv[city_col] = _clean_city(df_city_conv[city_col])
#             df_city_conv = df_city_conv.dropna(subset=[city_col])

#             sess_city = (
#                 df_city_conv.groupby([city_col, session_col])[cart_col]
#                             .any().reset_index()
#             )
#             conv_city = (
#                 sess_city.groupby(city_col)[cart_col]
#                          .mean().reset_index(name="conversion")
#             )
#             conv_city = conv_city[conv_city[city_col].isin(order_cities)]
#             conv_city[city_col] = pd.Categorical(conv_city[city_col],
#                                                  categories=order_cities, ordered=True)
#             conv_city = conv_city.sort_values(city_col)

#             fig_city_conv = px.bar(
#                 conv_city, x=city_col, y="conversion",
#                 title="도시별 전환율 (세션→장바구니, %)",
#                 color_discrete_sequence=[GOOGLE_BLUE]
#             )
#             fig_city_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
#             fig_city_conv.update_layout(yaxis_tickformat=".1%")
#             fig_city_conv.update_xaxes(categoryorder="array", categoryarray=order_cities)
#             st.plotly_chart(fig_city_conv, use_container_width=True, key=ukey("city-conv"))
#         else:
#             st.info("표시할 도시 데이터가 없습니다.")
#     else:
#         st.warning("도시 컬럼(city/City/regionCity)을 찾을 수 없습니다. (로더 화이트리스트 확인)")

#     # ---- 고착도: 탭 클릭(일/주) — 도시 유무와 무관하게 항상 표시 ----
#     d0 = dfp[[date_col, user_col]].drop_duplicates().copy()
#     d0[date_col] = pd.to_datetime(d0[date_col], errors="coerce")
#     d0 = d0[d0[date_col].notna()]
#     d0["date"]  = d0[date_col].dt.normalize()
#     d0["week"]  = d0[date_col].dt.to_period("W").dt.start_time
#     d0["month"] = d0[date_col].dt.to_period("M").dt.start_time

#     tab_day, tab_week = st.tabs(["일별 고착도 (WAU/DAU)", "주별 고착도 (MAU/WAU)"])

#     with tab_day:
#         if d0.empty:
#             st.info("고착도 계산을 위한 데이터가 부족합니다.")
#         else:
#             daily = d0.groupby("date")[user_col].nunique().rename("DAU").reset_index()
#             # 날짜별 7일 rolling window로 WAU 계산
#             dates_sorted = daily["date"].sort_values()
#             wau = []
#             for d in dates_sorted:
#                 win = d0[(d0["date"] >= (d - pd.Timedelta(days=6))) & (d0["date"] <= d)]
#                 wau.append({"date": d, "WAU": win[user_col].nunique()})
#             stick_daily = daily.merge(pd.DataFrame(wau), on="date", how="left")
#             stick_daily["WAU/DAU"] = (stick_daily["WAU"] / stick_daily["DAU"]).replace([np.inf, -np.inf], np.nan)

#             fig_stick_daily = px.line(
#                 stick_daily, x="date", y="WAU/DAU",
#                 title="고착도 (WAU/DAU, 일별)"
#             )
#             st.plotly_chart(fig_stick_daily, use_container_width=True, key=ukey("stick-wau-dau-citytab"))

#     with tab_week:
#         if d0.empty:
#             st.info("고착도 계산을 위한 데이터가 부족합니다.")
#         else:
#             weekly  = d0.groupby("week")[user_col].nunique().rename("WAU").reset_index()
#             monthly = d0.groupby("month")[user_col].nunique().rename("MAU").reset_index()
#             weekly["month"] = pd.to_datetime(weekly["week"]).dt.to_period("M").dt.start_time
#             wk = weekly.merge(monthly, on="month", how="left")
#             wk["MAU/WAU"] = (wk["MAU"] / wk["WAU"]).replace([np.inf, -np.inf], np.nan)

#             fig_stick_weekly = px.line(
#                 wk, x="week", y="MAU/WAU",
#                 title="고착도 (MAU/WAU, 주별)"
#             )
#             st.plotly_chart(fig_stick_weekly, use_container_width=True, key=ukey("stick-mau-wau-citytab"))






# # =========================================================
# # 5) Retention — 탭(클릭) 방식 + 한 번에 30/90일 계산
# # =========================================================
# st.header("3. Retention")

# # 코호트 준비 (유저-날짜 중복 제거 후 정렬)
# cohort = (
#     dfp[[user_col, cluster_col, date_col]]
#       .drop_duplicates()
#       .sort_values([user_col, date_col])
# )

# # 대표 클러스터(기간 내 최빈값) — 기존에 rep_cluster가 있으면 재사용
# def _rep_mode(s: pd.Series):
#     m = pd.to_numeric(s, errors="coerce")
#     m = m[m.notna()]
#     return str(int(m.mode().iat[0])) if not m.mode().empty else None

# rep_cluster_mode = (
#     cohort.groupby(user_col)[cluster_col]
#           .apply(_rep_mode)
#           .reset_index(name=cluster_col)
# )

# # 첫 방문일 계산
# first_visit = (
#     cohort.groupby(user_col, as_index=False)[date_col]
#           .min()
#           .rename(columns={date_col: "first_visit"})
# )

# # 유저-방문 레벨에 첫 방문일 붙여서, 30/90일 재방문 여부를 한 번에 계산
# cohort2 = cohort.merge(first_visit, on=user_col, how="left")
# def _ret_flag(days: int) -> pd.Series:
#     limit = cohort2["first_visit"] + pd.to_timedelta(days, "D")
#     flag = (cohort2[date_col] > cohort2["first_visit"]) & (cohort2[date_col] <= limit)
#     return cohort2.assign(flag=flag).groupby(user_col)["flag"].any().rename(f"ret_{days}")

# ret30 = _ret_flag(30)
# ret90 = _ret_flag(90)

# # 유저 단위 테이블(최초 방문 + 30/90일 재방문 + 코호트월 + 대표클러스터)
# ur = (
#     first_visit
#       .merge(ret30.reset_index(), on=user_col, how="left")
#       .merge(ret90.reset_index(), on=user_col, how="left")
#       .merge(rep_cluster_mode, on=user_col, how="left")
# )
# ur["cohort_month"] = ur["first_visit"].dt.to_period("M").dt.start_time

# def render_retention_heatmap(ret_col: str, window_label: str, key_suffix: str):
#     pivot = (
#         ur.groupby(["cohort_month", cluster_col])[ret_col]
#           .mean().reset_index()
#           .pivot(index="cohort_month", columns=cluster_col, values=ret_col)
#     )
#     pivot.columns = pivot.columns.astype(str)
#     col_order = [c for c in DESIRED_ORDER if c in pivot.columns]
#     if col_order:
#         pivot = pivot[col_order]

#     z = pivot.values
#     x = list(pivot.columns)
#     y = [pd.to_datetime(d).strftime("%Y-%m") for d in pivot.index]
#     text = np.where(np.isnan(z), "", (z*100).round(1).astype(str) + "%")

#     fig = go.Figure(data=go.Heatmap(
#         z=z, x=x, y=y, colorscale="Blues",
#         hovertemplate="Cohort %{y}<br>Cluster %{x}<br>Retention %{z:.1%}<extra></extra>"
#     ))
#     fig.update_layout(title=f"클러스터별 코호트 유지율 Heatmap ({window_label}) — (열=Cluster, 행=Cohort Month)")
#     fig.add_trace(go.Scatter(
#         x=np.repeat(x, len(y)), y=np.tile(y, len(x)),
#         mode="text", text=text.flatten(), hoverinfo="skip", showlegend=False
#     ))
#     st.plotly_chart(fig, use_container_width=True, key=ukey(f"ret-heatmap-{key_suffix}"))

# # 👇 탭(클릭) 방식: 로딩 없이 전환
# tab30, tab90 = st.tabs(["30일", "90일"])
# with tab30:
#     render_retention_heatmap("ret_30", "30일", "30")
# with tab90:
#     render_retention_heatmap("ret_90", "90일", "90")


# # Cart usage (user-level)
# cart_usage = (uf.groupby(cluster_col)["cart_any"]
#               .mean().reset_index(name="cart_usage_rate"))
# cart_usage[cluster_col] = pd.Categorical(cart_usage[cluster_col].astype(str),
#                                          categories=DESIRED_ORDER, ordered=True)
# cart_usage = cart_usage.sort_values(cluster_col)

# fig_cu = px.bar(
#     cart_usage, x=cluster_col, y="cart_usage_rate",
#     title="클러스터별 장바구니 이용률 (유저 기준)",
#     color=cluster_col, color_discrete_map=CLUSTER_COLORS,
#     labels={"cart_usage_rate":"장바구니 이용률", cluster_col:"Cluster"}
# )
# fig_cu.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
# fig_cu.update_layout(yaxis_tickformat=".1%")
# st.plotly_chart(fig_cu, use_container_width=True, key=ukey("ret-cart-usage"))


# st.caption("※ 분모=클러스터 인원은 유저 최신 스냅샷(중복 제거)으로 계산합니다. 퍼널/리텐션은 기간 전체 any 기준. 채널은 trafficMedium→trafficMed→trafficSource 우선 사용.")

# =========================================================
# 1) 데이터 로더: 업로드 CSV/ZIP 또는 리포지토리 내 CSV/ZIP (메모리 절약형)
# =========================================================
from zipfile import ZipFile
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

REPO_CSV = Path(__file__).parent / "google_with_cluster3_aha.csv"
REPO_ZIP = Path(__file__).parent / "google_with_cluster3_aha.zip"

st.sidebar.header("데이터")
uploaded = st.sidebar.file_uploader("CSV 또는 ZIP 업로드(선택)", type=["csv", "zip"])

# ---- 메모리 절약 설정 ----
st.sidebar.subheader("메모리 절약 설정")
lite_mode = st.sidebar.checkbox("메모리 절약 모드(샘플링)", value=True)
n_rows = st.sidebar.number_input("읽을 최대 행 수 (샘플링)", min_value=10_000, max_value=5_000_000,
                                 step=10_000, value=500_000)
downcast_obj_to_cat = st.sidebar.checkbox("문자열을 category로 변환", value=True)

# 앱에서 실제로 사용하는 컬럼만 지정
USECOLS = [
    "fullVisitorId","date","cluster",
    "visitStartTime","totalTimeOnSite","totalPageviews",
    "productPagesViewed","addedToCart","AhaMoment",
    "isFirstVisit","deviceCategory","country",
    "trafficMedium","trafficMed","trafficSource",
    "city","City","regionCity"
]

# 더 작은 dtype으로 지정
DTYPES = {
    "fullVisitorId": "string",
    "cluster": "Int8",
    "visitStartTime": "Int64",
    "totalTimeOnSite": "float32",
    "totalPageviews": "float32",
    "productPagesViewed": "float32",
    "addedToCart": "float32",   # 나중에 bool 로직 적용
    "isFirstVisit": "Int8",
}

def _read_csv(file_like_or_path):
    """필요 컬럼만, 작은 dtype으로, (옵션) 샘플링해서 읽기"""
    # pyarrow 가 있으면 속도/메모리 유리, 없으면 pandas 기본 엔진
    engine = "pyarrow"
    try:
        import pyarrow  # noqa: F401
    except Exception:
        engine = "python"

    return pd.read_csv(
        file_like_or_path,
        usecols=lambda c: c in USECOLS,
        dtype=DTYPES,
        parse_dates=["date"],
        infer_datetime_format=True,
        dayfirst=False,
        nrows=(int(n_rows) if lite_mode else None),
        engine=engine,
        on_bad_lines="skip",
        low_memory=False,
    )

def _read_from_zip(src):
    with ZipFile(src) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("ZIP 안에서 CSV 파일을 찾지 못했습니다.")
        with zf.open(names[0]) as f:
            return _read_csv(f)

# ---- 우선순위: 업로드 > 리포지토리 CSV > 리포지토리 ZIP ----
if uploaded is not None:
    if uploaded.name.lower().endswith(".csv"):
        df = _read_csv(uploaded)
        st.success("업로드한 CSV를 불러왔습니다.")
    else:
        df = _read_from_zip(uploaded)
        st.success("업로드한 ZIP에서 CSV를 불러왔습니다.")
elif REPO_CSV.exists():
    df = _read_csv(REPO_CSV)
    st.info(f"리포지토리 CSV에서 로드: {REPO_CSV.name}")
elif REPO_ZIP.exists():
    df = _read_from_zip(REPO_ZIP)
    st.info(f"리포지토리 ZIP에서 로드: {REPO_ZIP.name}")
else:
    st.error("데이터 소스를 찾을 수 없습니다. CSV/ZIP을 업로드하거나 리포지토리에 추가하세요.")
    st.stop()

# ---- 추가 다운캐스트/정리 ----
# 숫자 downcast
for c in df.select_dtypes(include=["float64"]).columns:
    df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
for c in df.select_dtypes(include=["int64"]).columns:
    df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")

# 도시 컬럼 표준화 (city 하나로)
if "City" in df.columns and "city" not in df.columns:
    df = df.rename(columns={"City": "city"})
if "regionCity" in df.columns and "city" not in df.columns:
    df = df.rename(columns={"regionCity": "city"})

# 문자열 → category (메모리 절감)
if downcast_obj_to_cat:
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        if c not in {"date", "fullVisitorId"}:
            df[c] = df[c].astype("category")

if df.empty:
    st.error("빈 데이터입니다."); st.stop()


# =========================================================
# 2) 공통 유틸/전처리 (형 변환 등)
# =========================================================
def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    try:
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0)
    except Exception:
        ss = s.astype(str).str.lower().str.strip()
        return ss.isin({"1", "true", "t", "y", "yes"})

def ensure_dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if is_datetime64tz_dtype(out):
        try:
            out = out.dt.tz_convert("UTC").dt.tz_localize(None)
        except AttributeError:
            out = out.dt.tz_localize(None)
    return out

# 컬럼 이름 준비
user_col     = "fullVisitorId"
date_col     = "date"
cluster_col  = "cluster"
detail_col   = "productPagesViewed"
cart_col     = "addedToCart"
device_col   = "deviceCategory"
country_col  = "country"

# 채널 컬럼 자동 선택
if "trafficMedium" in df.columns:
    medium_col = "trafficMedium"
elif "trafficMed" in df.columns:
    medium_col = "trafficMed"
elif "trafficSource" in df.columns:
    medium_col = "trafficSource"
else:
    medium_col = None

# 타입 정리 (가능한 한 견고하게)
if user_col in df.columns:
    df[user_col] = df[user_col].astype("string").fillna("")
if date_col in df.columns:
    df[date_col] = ensure_dt(df[date_col])
if detail_col in df.columns:
    df[detail_col] = pd.to_numeric(df[detail_col], errors="coerce").fillna(0)
if cart_col in df.columns:
    df[cart_col] = to_bool(df[cart_col])
if "AhaMoment" in df.columns:
    df["AhaMoment"] = to_bool(df["AhaMoment"])
if cluster_col in df.columns:
    cnum = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")
    df = df[cnum.isin([0, 1, 2])].copy()
    df[cluster_col] = cnum.astype("Int64").astype(str)

# date 필터를 위한 검증
if date_col not in df.columns or df[date_col].notna().sum() == 0:
    st.error("`date` 컬럼이 없거나 형식이 잘못되어 있습니다."); st.stop()
df = df[df[date_col].notna()].copy()

# 세션 surrogate (있으면 30분 버킷, 없으면 날짜 단위)
session_col = "_session_surrogate"
if df[date_col].dt.time.astype(str).nunique() > 1:
    bucket = df[date_col].dt.floor("30min").astype(str)
else:
    bucket = df[date_col].dt.strftime("%Y-%m-%d")
if user_col in df.columns:
    df[session_col] = (df[user_col].astype(str) + "|" + bucket)
else:
    df[session_col] = bucket  # fallback


# =========================================================
# 3) 헤더/타이틀
# =========================================================
st.markdown(
    """
    <h1 style="font-size:40px; font-weight:bold;">
        <span style="color:#4285F4;">G</span>
        <span style="color:#DB4437;">o</span>
        <span style="color:#F4B400;">o</span>
        <span style="color:#4285F4;">g</span>
        <span style="color:#0F9D58;">l</span>
        <span style="color:#DB4437;">e</span>
        Merchandise Sale Prediction
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>", unsafe_allow_html=True)
st.title("Overview")


# =========================================================
# 4) Overview (세션/유저/KPI)
#    ※ 기존의 '윈도우 경로'로 CSV 읽던 부분을 모두 제거
# =========================================================

# visitStartTime이 있으면 세션 ID 생성에 활용 (없어도 동작)
if {"fullVisitorId", "visitStartTime"}.issubset(df.columns):
    df["sessionId"] = df["fullVisitorId"].astype(str) + "_" + df["visitStartTime"].astype(str)
else:
    df["sessionId"] = df.get(session_col, pd.Series(range(len(df)))).astype(str)

total_sessions = df["sessionId"].nunique()
unique_users = df[user_col].nunique() if user_col in df.columns else np.nan
sessions_per_user = (total_sessions / unique_users) if unique_users else 0

c1, c2, c3 = st.columns(3)
c1.metric("총 세션 수", f"{total_sessions:,}")
c2.metric("총 유저 수", f"{unique_users:,}" if pd.notna(unique_users) else "NA")
c3.metric("유저당 평균 방문 횟수", f"{sessions_per_user:.2f}" if unique_users else "NA")

# 평균 체류시간 / 페이지뷰
avg_time_minutes = (pd.to_numeric(df.get("totalTimeOnSite"), errors="coerce").mean() or 0) / 60
avg_pageviews = pd.to_numeric(df.get("totalPageviews"), errors="coerce").mean() or 0
c1, c2 = st.columns(2)
c1.metric("평균 체류시간 (분)", f"{avg_time_minutes:.1f}")
c2.metric("세션당 평균 페이지뷰", f"{avg_pageviews:.2f}")

# 신규/재방문 비율 + 카트전환율
if "isFirstVisit" in df.columns and cart_col in df.columns:
    df["visit_type"] = np.where(pd.to_numeric(df["isFirstVisit"], errors="coerce") == 1, "신규", "재방문")
    added_num = pd.to_numeric(df["addedToCart"], errors="coerce")
    added_str = df["addedToCart"].astype(str).str.lower()
    df["cart_flag"] = (added_num.fillna(0) > 0) | (added_str.isin(["1", "true", "t", "y", "yes"]))

    summary = (
        df.groupby("visit_type")
          .agg(sessions=("fullVisitorId", "count"), cart_sessions=("cart_flag", "sum"))
          .reset_index()
    )
    summary["cart_rate"] = summary["cart_sessions"] / summary["sessions"]

    st.subheader("신규/재방문 회원별 Cart 전환율")
    left, right = st.columns([1, 1])
    with left:
        summary_view = summary.copy()
        summary_view["session_ratio(%)"] = (
            summary_view["sessions"] / summary_view["sessions"].sum() * 100
        ).round(1).astype(str) + "%"
        summary_view["cart_rate(%)"] = (summary_view["cart_rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(summary_view[["visit_type", "session_ratio(%)", "cart_rate(%)"]], use_container_width=True)
    with right:
        fig = px.bar(
            summary,
            x="visit_type",
            y="cart_rate",
            text=(summary["cart_rate"] * 100).round(1).astype(str) + "%",
            labels={"visit_type": "방문 유형", "cart_rate": "Cart 전환율"},
            category_orders={"visit_type": ["신규", "재방문"]},
            color="visit_type",
            color_discrete_map={"신규": "#4285F4", "재방문": "#F4B400"},
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(textposition="outside")
        fig.update_layout(bargap=0.7, bargroupgap=0.1)
        st.plotly_chart(fig, use_container_width=True)

# AHA 달성률
if "AhaMoment" in df.columns:
    s = df["AhaMoment"]
    aha = s if pd.api.types.is_bool_dtype(s) else (
        (pd.to_numeric(s, errors="coerce").fillna(0) > 0) |
        (s.astype(str).str.strip().str.lower().isin(["true", "t", "y", "yes", "1"]))
    )
    aha_rate = aha.mean()
    st.subheader("전체 AHA 달성률")
    st.metric("Aha 달성률", f"{aha_rate*100:.1f}%")
    pie_df = pd.DataFrame({"Aha": ["TRUE", "FALSE"], "count": [int(aha.sum()), int((~aha).sum())]})
    fig = px.pie(
        pie_df, names="Aha", values="count", hole=0.35, color="Aha",
        color_discrete_map={"TRUE": "#F4B400", "FALSE": "#4285F4"}
    )
    fig.update_traces(textposition="inside", texttemplate="%{label}: %{percent:.1%}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(
            """
            **AHA 모먼트 측정 기준**  
            - 처음으로 상품을 장바구니에 담았을 때  
            - 세션 시간(TimePerSession)이 일정 기준(예: 3 이상)에 도달했을 때
            """
        )

# 퍼널 분석 (세션 기준)
st.subheader("퍼널 분석")
step1 = (pd.to_numeric(df.get("totalPageviews"), errors="coerce").fillna(0) >= 1)
step2 = (pd.to_numeric(df.get("productPagesViewed"), errors="coerce").fillna(0) >= 1)
step3 = (pd.to_numeric(df.get("addedToCart"), errors="coerce").fillna(0) >= 1)
n1, n2, n3 = int(step1.sum()), int((step1 & step2).sum()), int((step1 & step2 & step3).sum())
conv12 = n2 / n1 if n1 else 0
conv23 = n3 / n2 if n2 else 0
conv_overall = n3 / n1 if n1 else 0
k1, k2, k3 = st.columns(3)
k1.metric("페이지뷰→제품상세 전환율", f"{conv12*100:.1f}%")
k2.metric("제품상세→카트추가 전환율", f"{conv23*100:.1f}%")
k3.metric("페이지뷰→카트추가 전환율", f"{conv_overall*100:.1f}%")

funnel_df = pd.DataFrame({"Step": ["페이지뷰 ≥1", "제품상세 ≥1", "카트추가 ≥1"], "Sessions": [n1, n2, n3]})
pct_prev = [1.0, (n2 / n1) if n1 else 0, (n3 / n2) if n2 else 0]
funnel_df["Text"] = [f"{v:,.0f} ({p*100:.1f}%)" for v, p in zip(funnel_df["Sessions"], pct_prev)]
fig_funnel = px.funnel(funnel_df, x="Sessions", y="Step")
fig_funnel.update_traces(text=funnel_df["Text"], texttemplate="%{text}", textposition="inside",
                         textinfo="none", marker_color="#4285F4")
st.plotly_chart(fig_funnel, use_container_width=True)

st.markdown("<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>", unsafe_allow_html=True)


# =========================================================
# 5) Cluster Summary Table (샘플)
# =========================================================
st.title("Cluster Summary Table")
data = {
    "고객군": ["탐색형 고객 (Explorers)", "비활성 고객 (Visitors)", "충성/핵심 고객 (Core Buyers)"],
    "유저 수": ["260,180", "305,542", "29,126"],
    "Recency (일)": [159.0, 164.4, 150.4],
    "Frequency (방문일수)": [1.21, 1.07, 1.75],
    "Cart Rate": [0.01, 0.00, 0.77],
    "Search Rate": [0.96, 0.01, 0.95],
    "Time/Session (정규화)": [1.19, 0.01, 5.32],
}
df_cluster = pd.DataFrame(data)
st.subheader("클러스터별 요약")
st.dataframe(df_cluster, use_container_width=True)

st.subheader("클러스터별 유저 비중")
data = {
    "Cluster": [0, 1, 2],
    "클러스터링 결과": ["탐색형 고객 (Explorers)", "비활성 고객 (Visitors)", "충성/핵심 고객 (Core Buyers)"],
    "유저 수": [260180, 305542, 29126],
}
df_cluster = pd.DataFrame(data)
df_cluster["label"] = df_cluster["클러스터링 결과"].str.strip()
order = ["탐색형 고객 (Explorers)", "비활성 고객 (Visitors)", "충성/핵심 고객 (Core Buyers)"]
colors = ["#4285F4", "#F4B400", "#DB4437"]
fig = px.pie(
    df_cluster, names="label", values="유저 수", hole=0.3,
    category_orders={"label": order}, color="label", color_discrete_sequence=colors
)
fig.update_traces(textinfo="label+percent", pull=[0.02, 0.02, 0.05], rotation=90)
fig.update_layout(margin=dict(t=40, b=40, l=40, r=40),
                  legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>", unsafe_allow_html=True)


# =========================================================
# 6) AARRR Dashboard — USER level (원본 로직 유지, 로더/설정 중복 제거)
# =========================================================
st.title("AARRR Dashboard — USER level")

GOOGLE_BLUE   = "#4285F4"
GOOGLE_YELLOW = "#F4B400"
GOOGLE_RED    = "#DB4437"
GOOGLE_GREY   = "#9AA0A6"
CLUSTER_COLORS = {"0": GOOGLE_BLUE, "1": GOOGLE_YELLOW, "2": GOOGLE_RED}
DESIRED_ORDER = ["0", "1", "2"]
def ukey(name): return f"{name}-{uuid.uuid4().hex[:8]}"

# 날짜 필터
st.sidebar.header("기간")
min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
start = st.sidebar.date_input("시작일", value=min_d)
end   = st.sidebar.date_input("종료일", value=max_d)
start_dt = pd.Timestamp(start)
end_dt   = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
dfp = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
if dfp.empty:
    st.warning("선택 구간 데이터 없음"); st.stop()

# 스냅샷 & any 플래그
def rep_mode(s: pd.Series) -> Optional[str]:
    m = pd.to_numeric(s, errors="coerce"); m = m[m.notna()]
    if m.empty: return None
    return str(int(m.mode().iat[0]))

df_user_last = (
    dfp.sort_values([user_col, date_col])
       .drop_duplicates(subset=[user_col], keep="last")
       [[user_col, cluster_col, "AhaMoment", cart_col]]
       .copy()
)
df_user_last[cluster_col] = df_user_last[cluster_col].astype(str)
df_user_last["AhaMoment"] = to_bool(df_user_last["AhaMoment"])
df_user_last["cart_bool"] = to_bool(df_user_last[cart_col])

rep_cluster_mode = (dfp.groupby(user_col)[cluster_col].apply(rep_mode)
                       .reset_index(name=cluster_col))
user_any = (dfp.groupby(user_col)
              .agg(detail_any=(detail_col, lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).any())),
                   cart_any=(cart_col, "any"))
              .reset_index())
uf = user_any.merge(rep_cluster_mode, on=user_col, how="left")
uf["home"] = 1

# 6-1) Activation
st.header("1. Activation (분모=클러스터 인원, 유저 최신 스냅샷)")
cluster_pop = (df_user_last.groupby(cluster_col)[user_col].nunique().rename("cluster_users").reset_index())
cluster_pop[cluster_col] = pd.Categorical(cluster_pop[cluster_col], categories=DESIRED_ORDER, ordered=True)
cluster_pop = cluster_pop.sort_values(cluster_col)
aha_true = (df_user_last.groupby(cluster_col)["AhaMoment"].sum().rename("aha_true").reset_index())
aha_rate_den_cluster = (cluster_pop.merge(aha_true, on=cluster_col, how="left")
                        .fillna({"aha_true":0})
                        .assign(aha_rate=lambda t: t["aha_true"] / t["cluster_users"].replace(0, np.nan))
                        [[cluster_col,"cluster_users","aha_true","aha_rate"]])
aha_rate_den_cluster[cluster_col] = pd.Categorical(aha_rate_den_cluster[cluster_col],
                                                   categories=DESIRED_ORDER, ordered=True)
aha_rate_den_cluster = aha_rate_den_cluster.sort_values(cluster_col)

aha_cart_true = ((df_user_last["AhaMoment"] & df_user_last["cart_bool"])
                 .groupby(df_user_last[cluster_col]).sum()
                 .rename("aha_cart_true").reset_index())
aha_cart_overall_den_cluster = (cluster_pop.merge(aha_cart_true, on=cluster_col, how="left")
                                .fillna({"aha_cart_true":0})
                                .assign(aha_cart_overall=lambda t: t["aha_cart_true"] / t["cluster_users"].replace(0, np.nan))
                                [[cluster_col,"cluster_users","aha_cart_true","aha_cart_overall"]])
aha_cart_overall_den_cluster[cluster_col] = pd.Categorical(aha_cart_overall_den_cluster[cluster_col],
                                                           categories=DESIRED_ORDER, ordered=True)
aha_cart_overall_den_cluster = aha_cart_overall_den_cluster.sort_values(cluster_col)

fig_aha = px.bar(
    aha_rate_den_cluster, y=cluster_col, x="aha_rate", orientation="h",
    title="클러스터별 Aha 달성률 (분모=클러스터 인원)",
    color=cluster_col, color_discrete_map=CLUSTER_COLORS,
    labels={"aha_rate":"Aha 달성률", cluster_col:"Cluster"}
)
fig_aha.update_yaxes(categoryorder="array", categoryarray=DESIRED_ORDER)
fig_aha.update_traces(texttemplate="%{x:.1%}", textposition="outside", showlegend=False)
fig_aha.update_layout(xaxis_tickformat=".1%")
st.plotly_chart(fig_aha, use_container_width=True, key=ukey("act-aha"))

fig_aha_cart = px.bar(
    aha_cart_overall_den_cluster, y=cluster_col, x="aha_cart_overall", orientation="h",
    title="클러스터별 Aha → Cart (분모=클러스터 인원)",
    color=cluster_col, color_discrete_map=CLUSTER_COLORS,
    labels={"aha_cart_overall":"Aha→Cart (전체 대비)", cluster_col:"Cluster"}
)
fig_aha_cart.update_yaxes(categoryorder="array", categoryarray=DESIRED_ORDER)
fig_aha_cart.update_traces(texttemplate="%{x:.1%}", textposition="outside", showlegend=False)
fig_aha_cart.update_layout(xaxis_tickformat=".1%")
st.plotly_chart(fig_aha_cart, use_container_width=True, key=ukey("act-aha-cart"))

st.dataframe(
    aha_rate_den_cluster.merge(
        aha_cart_overall_den_cluster[[cluster_col,"aha_cart_true","aha_cart_overall"]],
        on=cluster_col, how="left"
    ).rename(columns={
        "cluster_users":"클러스터 인원",
        "aha_true":"Aha True 수",
        "aha_rate":"Aha 달성률",
        "aha_cart_true":"Aha∩Cart True 수",
        "aha_cart_overall":"Aha→Cart (전체 대비)"
    })[[cluster_col,"클러스터 인원","Aha True 수","Aha 달성률","Aha∩Cart True 수","Aha→Cart (전체 대비)"]],
    use_container_width=True
)

# 6-2) Funnel (USER 기준)
st.subheader("퍼널 전환율")
tabs = st.tabs(["Cluster 0", "Cluster 1", "Cluster 2"])
tab_keys = ["0", "1", "2"]

def draw_home_based_funnel(uv: pd.DataFrame, title: str, key_sfx: str):
    h = int(len(uv))
    d = int((uv["detail_any"] == 1).sum())
    k = int(((uv["detail_any"] == 1) & (uv["cart_any"] == 1)).sum())
    pct_detail = (d / h) if h else np.nan
    pct_cart   = (k / h) if h else np.nan
    seg_text = [f"{h:,}\n100%", f"{d:,}\n{pct_detail:.1%} (Home→Detail)", f"{k:,}\n{pct_cart:.1%} (Home→Cart via Detail)"]
    fig = go.Figure(go.Funnel(
        y=["Home","Detail","Cart(Detail 경유)"], x=[h, d, k],
        text=seg_text, textposition="inside", textinfo="text",
        marker={"color": [GOOGLE_BLUE, GOOGLE_BLUE, GOOGLE_BLUE]},
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True, key=ukey(f"funnel-{key_sfx}"))
    st.dataframe(pd.DataFrame([{
        "home_users": h, "detail_users": d, "cart_users": k,
        "Home→Detail(%)": round((pct_detail * 100), 1) if h else np.nan,
        "Home→Cart(Detail 경유)(%)": round((pct_cart * 100), 1) if h else np.nan,
        "Detail→Cart(%)": round(((k / d) * 100), 1) if d else np.nan,
    }]), use_container_width=True)

for t, key in zip(tabs, tab_keys):
    with t:
        uv = uf[uf[cluster_col] == key].copy()
        draw_home_based_funnel(uv, f"Funnel (Cluster {key})", key)

# 6-3) Acquisition (USER 기준)
st.header("2. Acquisition (USER 기준)")
cA, cB = st.columns([1,1])
with cA:
    min_share = st.slider("최소 유입 비중 제외", 0.0, 0.2, 0.01, 0.005, key=ukey("acq-minshare"))
with cB:
    top_n = st.number_input("TOP N", min_value=3, max_value=30, value=10, step=1, key=ukey("acq-topn"))

def _clean_series_exclude_noise(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    bad = {"", "nan", "(none)", "none", "(not set)", "not set", "unavailable"}
    return ss.where(~ss.str.lower().isin(bad), other=np.nan)

def _clean_channel_keep_none(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    bad = {"", "nan", "(not set)", "not set", "unavailable"}
    return ss.where(~ss.str.lower().isin(bad), other=np.nan)

def render_acquisition_for_dim(dfp_src: pd.DataFrame, dim_col: str, title_prefix: str):
    if dim_col not in dfp_src.columns:
        st.info(f"`{dim_col}` 컬럼이 없어 건너뜁니다.")
        return
    dfp_acq = dfp_src.copy()
    rep_dim = (dfp_acq[[user_col, dim_col, date_col]]
               .sort_values([user_col, date_col])
               .drop_duplicates(user_col, keep="last")[[user_col, dim_col]])
    rep_dim[dim_col] = _clean_series_exclude_noise(rep_dim[dim_col])
    acq = (rep_dim.dropna(subset=[dim_col])[dim_col]
           .value_counts(normalize=True)
           .rename_axis(dim_col).reset_index(name="share").sort_values("share", ascending=False))
    acq = acq[acq["share"] >= float(min_share)].head(int(top_n))
    order = acq[dim_col].tolist()
    fig_share = px.bar(acq, x=dim_col, y="share", title=f"{title_prefix} 상위 {len(acq)} 유입 비중 (USER)",
                       color_discrete_sequence=[GOOGLE_BLUE])
    fig_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
    fig_share.update_layout(yaxis_tickformat=".1%")
    fig_share.update_xaxes(categoryorder="array", categoryarray=order)
    st.plotly_chart(fig_share, use_container_width=True, key=ukey(f"acq-share-{dim_col}"))

    sess_cart = (dfp_acq.dropna(subset=[dim_col])
                 .groupby([dim_col, session_col])[cart_col].any().reset_index())
    conv = (sess_cart.groupby(dim_col)[cart_col].mean().reset_index(name="conversion"))
    conv = conv[conv[dim_col].isin(order)]
    conv[dim_col] = pd.Categorical(conv[dim_col], categories=order, ordered=True)
    conv = conv.sort_values(dim_col)
    fig_conv = px.bar(conv, x=dim_col, y="conversion", title=f"{title_prefix}별 전환율 (세션→장바구니, %)",
                      color_discrete_sequence=[GOOGLE_BLUE])
    fig_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
    fig_conv.update_layout(yaxis_tickformat=".1%")
    fig_conv.update_xaxes(categoryorder="array", categoryarray=order)
    st.plotly_chart(fig_conv, use_container_width=True, key=ukey(f"acq-conv-{dim_col}"))

tab_channel, tab_device, tab_country, tab_city = st.tabs(["채널", "디바이스", "국가", "도시"])

with tab_channel:
    st.subheader("채널 (trafficMedium/trafficMed/trafficSource) — (none 포함)")
    if medium_col is None or medium_col not in dfp.columns:
        st.info("채널 컬럼이 없어 건너뜁니다.")
    else:
        rep_medium = (dfp[[user_col, medium_col, date_col]]
                      .sort_values([user_col, date_col])
                      .drop_duplicates(user_col, keep="last")[[user_col, medium_col]])
        rep_medium[medium_col] = _clean_channel_keep_none(rep_medium[medium_col])
        channel_share = (rep_medium.dropna(subset=[medium_col])[medium_col]
                         .value_counts(normalize=True).rename_axis(medium_col)
                         .reset_index(name="share").sort_values("share", ascending=False))
        channel_share = channel_share[channel_share["share"] >= float(min_share)].head(int(top_n))
        order_channels = channel_share[medium_col].tolist()
        fig_ch_share = px.bar(channel_share, x=medium_col, y="share",
                              title=f"채널별 유입 비중 (USER) — Top {len(channel_share)}",
                              color_discrete_sequence=[GOOGLE_BLUE])
        fig_ch_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
        fig_ch_share.update_layout(yaxis_tickformat=".1%")
        fig_ch_share.update_xaxes(categoryorder="array", categoryarray=order_channels)
        st.plotly_chart(fig_ch_share, use_container_width=True, key=ukey("acq-channel-share"))

        df_conv = dfp.copy()
        df_conv[medium_col] = _clean_channel_keep_none(df_conv[medium_col])
        df_conv = df_conv.dropna(subset=[medium_col])
        sess_cart = (df_conv.groupby([medium_col, session_col])[cart_col].any().reset_index())
        conv = (sess_cart.groupby(medium_col)[cart_col].mean().reset_index(name="conversion"))
        conv = conv[conv[medium_col].isin(order_channels)]
        conv[medium_col] = pd.Categorical(conv[medium_col], categories=order_channels, ordered=True)
        conv = conv.sort_values(medium_col)
        fig_ch_conv = px.bar(conv, x=medium_col, y="conversion", title="채널별 전환율 (세션→장바구니, %)",
                             color_discrete_sequence=[GOOGLE_BLUE])
        fig_ch_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
        fig_ch_conv.update_layout(yaxis_tickformat=".1%")
        fig_ch_conv.update_xaxes(categoryorder="array", categoryarray=order_channels)
        st.plotly_chart(fig_ch_conv, use_container_width=True, key=ukey("acq-channel-conv"))

with tab_device:
    render_acquisition_for_dim(dfp, device_col, "디바이스")

with tab_country:
    render_acquisition_for_dim(dfp, country_col, "국가")

with tab_city:
    def _clean_city(s: pd.Series) -> pd.Series:
        ss = s.astype(str).str.strip()
        bad = {"", "nan", "(none)", "none", "(not set)", "not set", "unavailable", "not available in demo dataset"}
        return ss.where(~ss.str.lower().isin(bad), other=np.nan)

    possible_city_cols = ["city", "City", "regionCity"]
    city_col = next((c for c in possible_city_cols if c in dfp.columns), None)

    if city_col is not None:
        rep_city = (dfp[[user_col, city_col, date_col]]
                    .sort_values([user_col, date_col])
                    .drop_duplicates(user_col, keep="last")[[user_col, city_col]])
        rep_city[city_col] = _clean_city(rep_city[city_col])
        top_n_cities = int(top_n)
        city_share = (rep_city.dropna(subset=[city_col])[city_col]
                      .value_counts(normalize=True).rename_axis(city_col)
                      .reset_index(name="share").sort_values("share", ascending=False).head(top_n_cities))
        if not city_share.empty:
            order_cities = city_share[city_col].tolist()
            fig_city_share = px.bar(city_share, x=city_col, y="share", title="도시별 유입 비중 Top N (USER)",
                                    color_discrete_sequence=[GOOGLE_BLUE])
            fig_city_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
            fig_city_share.update_layout(yaxis_tickformat=".1%")
            fig_city_share.update_xaxes(categoryorder="array", categoryarray=order_cities)
            st.plotly_chart(fig_city_share, use_container_width=True, key=ukey("city-share"))

            df_city_conv = dfp.copy()
            df_city_conv[city_col] = _clean_city(df_city_conv[city_col])
            df_city_conv = df_city_conv.dropna(subset=[city_col])
            sess_city = (df_city_conv.groupby([city_col, session_col])[cart_col].any().reset_index())
            conv_city = (sess_city.groupby(city_col)[cart_col].mean().reset_index(name="conversion"))
            conv_city = conv_city[conv_city[city_col].isin(order_cities)]
            conv_city[city_col] = pd.Categorical(conv_city[city_col], categories=order_cities, ordered=True)
            conv_city = conv_city.sort_values(city_col)
            fig_city_conv = px.bar(conv_city, x=city_col, y="conversion", title="도시별 전환율 (세션→장바구니, %)",
                                   color_discrete_sequence=[GOOGLE_BLUE])
            fig_city_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
            fig_city_conv.update_layout(yaxis_tickformat=".1%")
            fig_city_conv.update_xaxes(categoryorder="array", categoryarray=order_cities)
            st.plotly_chart(fig_city_conv, use_container_width=True, key=ukey("city-conv"))
        else:
            st.info("표시할 도시 데이터가 없습니다.")
    else:
        st.warning("도시 컬럼(city/City/regionCity)을 찾을 수 없습니다. (데이터 컬럼 확인)")

    # 고착도 (일/주)
    d0 = dfp[[date_col, user_col]].drop_duplicates().copy()
    d0[date_col] = pd.to_datetime(d0[date_col], errors="coerce")
    d0 = d0[d0[date_col].notna()]
    d0["date"]  = d0[date_col].dt.normalize()
    d0["week"]  = d0[date_col].dt.to_period("W").dt.start_time
    d0["month"] = d0[date_col].dt.to_period("M").dt.start_time

    tab_day, tab_week = st.tabs(["일별 고착도 (WAU/DAU)", "주별 고착도 (MAU/WAU)"])
    with tab_day:
        if d0.empty:
            st.info("고착도 계산을 위한 데이터가 부족합니다.")
        else:
            daily = d0.groupby("date")[user_col].nunique().rename("DAU").reset_index()
            dates_sorted = daily["date"].sort_values()
            wau = []
            for d in dates_sorted:
                win = d0[(d0["date"] >= (d - pd.Timedelta(days=6))) & (d0["date"] <= d)]
                wau.append({"date": d, "WAU": win[user_col].nunique()})
            stick_daily = daily.merge(pd.DataFrame(wau), on="date", how="left")
            stick_daily["WAU/DAU"] = (stick_daily["WAU"] / stick_daily["DAU"]).replace([np.inf, -np.inf], np.nan)
            fig_stick_daily = px.line(stick_daily, x="date", y="WAU/DAU", title="고착도 (WAU/DAU, 일별)")
            st.plotly_chart(fig_stick_daily, use_container_width=True, key=ukey("stick-wau-dau-citytab"))

    with tab_week:
        if d0.empty:
            st.info("고착도 계산을 위한 데이터가 부족합니다.")
        else:
            weekly  = d0.groupby("week")[user_col].nunique().rename("WAU").reset_index()
            monthly = d0.groupby("month")[user_col].nunique().rename("MAU").reset_index()
            weekly["month"] = pd.to_datetime(weekly["week"]).dt.to_period("M").dt.start_time
            wk = weekly.merge(monthly, on="month", how="left")
            wk["MAU/WAU"] = (wk["MAU"] / wk["WAU"]).replace([np.inf, -np.inf], np.nan)
            fig_stick_weekly = px.line(wk, x="week", y="MAU/WAU", title="고착도 (MAU/WAU, 주별)")
            st.plotly_chart(fig_stick_weekly, use_container_width=True, key=ukey("stick-mau-wau-citytab"))

# 6-4) Retention (30/90일)
st.header("3. Retention")
cohort = (dfp[[user_col, cluster_col, date_col]].drop_duplicates().sort_values([user_col, date_col]))
def _rep_mode(s: pd.Series):
    m = pd.to_numeric(s, errors="coerce"); m = m[m.notna()]
    return str(int(m.mode().iat[0])) if not m.mode().empty else None
rep_cluster_mode = (cohort.groupby(user_col)[cluster_col].apply(_rep_mode).reset_index(name=cluster_col))
first_visit = (cohort.groupby(user_col, as_index=False)[date_col].min().rename(columns={date_col: "first_visit"}))
cohort2 = cohort.merge(first_visit, on=user_col, how="left")
def _ret_flag(days: int) -> pd.Series:
    limit = cohort2["first_visit"] + pd.to_timedelta(days, "D")
    flag = (cohort2[date_col] > cohort2["first_visit"]) & (cohort2[date_col] <= limit)
    return cohort2.assign(flag=flag).groupby(user_col)["flag"].any().rename(f"ret_{days}")
ret30 = _ret_flag(30); ret90 = _ret_flag(90)
ur = (first_visit.merge(ret30.reset_index(), on=user_col, how="left")
                .merge(ret90.reset_index(), on=user_col, how="left")
                .merge(rep_cluster_mode, on=user_col, how="left"))
ur["cohort_month"] = ur["first_visit"].dt.to_period("M").dt.start_time
def render_retention_heatmap(ret_col: str, window_label: str, key_suffix: str):
    pivot = (ur.groupby(["cohort_month", cluster_col])[ret_col]
             .mean().reset_index().pivot(index="cohort_month", columns=cluster_col, values=ret_col))
    pivot.columns = pivot.columns.astype(str)
    col_order = [c for c in DESIRED_ORDER if c in pivot.columns]
    if col_order: pivot = pivot[col_order]
    z = pivot.values; x = list(pivot.columns); y = [pd.to_datetime(d).strftime("%Y-%m") for d in pivot.index]
    text = np.where(np.isnan(z), "", (z*100).round(1).astype(str) + "%")
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale="Blues",
                                    hovertemplate="Cohort %{y}<br>Cluster %{x}<br>Retention %{z:.1%}<extra></extra>"))
    fig.update_layout(title=f"클러스터별 코호트 유지율 Heatmap ({window_label}) — (열=Cluster, 행=Cohort Month)")
    fig.add_trace(go.Scatter(x=np.repeat(x, len(y)), y=np.tile(y, len(x)),
                             mode="text", text=text.flatten(), hoverinfo="skip", showlegend=False))
    st.plotly_chart(fig, use_container_width=True, key=ukey(f"ret-heatmap-{key_suffix}"))
tab30, tab90 = st.tabs(["30일", "90일"])
with tab30: render_retention_heatmap("ret_30", "30일", "30")
with tab90: render_retention_heatmap("ret_90", "90일", "90")

# 장바구니 이용률
cart_usage = (uf.groupby(cluster_col)["cart_any"].mean().reset_index(name="cart_usage_rate"))
cart_usage[cluster_col] = pd.Categorical(cart_usage[cluster_col].astype(str), categories=DESIRED_ORDER, ordered=True)
cart_usage = cart_usage.sort_values(cluster_col)
fig_cu = px.bar(cart_usage, x=cluster_col, y="cart_usage_rate",
                title="클러스터별 장바구니 이용률 (유저 기준)",
                color=cluster_col, color_discrete_map=CLUSTER_COLORS,
                labels={"cart_usage_rate":"장바구니 이용률", cluster_col:"Cluster"})
fig_cu.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
fig_cu.update_layout(yaxis_tickformat=".1%")
st.plotly_chart(fig_cu, use_container_width=True, key=ukey("ret-cart-usage"))

st.caption("※ 분모=클러스터 인원은 유저 최신 스냅샷(중복 제거)으로 계산합니다. 퍼널/리텐션은 기간 전체 any 기준. 채널은 trafficMedium→trafficMed→trafficSource 우선 사용.")


