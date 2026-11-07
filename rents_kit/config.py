"""All settings in one place. Edit values here on the day; every module reads from this file.
Requirement ids (T1.1 ...) refer to requirements.md."""
from pathlib import Path

# ---- task rules ----------------------------------------------------------------------------
CUTOFF = "2026-07-31"                      # nothing dated after this may be used (US2)
TEAM_NAME = "team-name"                    # lower case, hyphens only (T1.6) - set on the day
CLEVER_STEP = ("We carry each flat's own past rent forward with a City of Sydney rent index "
               "built from repeat adverts, then measure the parking premium within the same building.")

# ---- validation (US3) ----------------------------------------------------------------------
TUNE_TRAIN_END, TUNE_TEST = "2026-04-01", ("2026-04-01", "2026-05-31")
CONFIRM_TRAIN_END, CONFIRM_TEST = "2026-06-01", ("2026-06-01", "2026-07-31")
SAMPLE_TRAIN, SAMPLE_TEST = ("2025-05-01", "2026-04-30"), ("2026-05-01", "2026-07-31")   # T3.5
SAMPLE_MIN_GROUP = 10
SAMPLE_REF = {"mdae": 100.0, "within100": 0.568, "r2": 0.645, "n": 3954}
SCORE_WITHIN = 100

# ---- data hygiene --------------------------------------------------------------------------
RENT_MIN, RENT_MAX = 100, 10_000           # weekly rent outside this range is dropped from training
CAMPAIGN_DAYS = 45                         # relists within this many days = one listing campaign (T9.4)
TRAIN_FROM = "2018-01-01"                  # training rows start here (older adverts still feed history)
BED_GROUPS_MAX = 4                         # bedroom groups 0,1,2,3,4+ (T7.6)
BED_FLAG_ABOVE = 6

# ---- rent index (US9) ----------------------------------------------------------------------
PAIR_MIN_GAP_DAYS = 60
PAIR_MAX_ABS_LOG = 0.5
INDEX_MIN_PAIRS = 150
INDEX_LAM2, INDEX_LAM1 = 3.0, 0.3
INDEX_END_MODE = "flat"                    # "flat" or "trend" (T9.3)
TREND_MONTHS = 6

# ---- features (US8) ------------------------------------------------------------------------
TIME_TAU_YEARS = 1.0
FLOOR_SCALE = 6.0
BED_SCALE = 1.0
SHRINK_K = 3                               # T8.5
VIEW_RADIUS_M = 100                        # T8.4
KNN_QUERY = 60
OWN_LAST_MAX_AGE_YEARS = 3

# ---- model ---------------------------------------------------------------------------------
LOSS = "absolute_error"
TARGET = "resid"                           # "resid" (learn log rent - anchor) or "direct"
HALFLIFE_YEARS = 1.5
SEEDS = [0, 1, 2, 3, 4]
USE_VIEWS = True                           # floors above neighbours as a model feature (T8.4)
BLEND_WEIGHT = None                        # set from eval output if the model does not beat own-last (T4.5)
CLIP_PCT = (0.1, 99.9)
ROUND_TO = 0.5                             # 0.5 (whole or .50), or 5 for nearest $5 (S7)

# ---- public features: G-NAF extras + LiDAR views (team repo microburbs-hackathon-2026) -------
USE_GNAF = True                            # floor/position/building size/age from Geoscape G-NAF
USE_LIDAR = True                           # per-floor views, heights, distances from LiDAR + OSM
GNAF_FLOOR_FILL = True                     # fill a blank organiser floor_level from the G-NAF floor (needs USE_GNAF)
USE_AIRBNB = False                         # Inside Airbnb counts of listings that existed before the cutoff (300 m, suburb);
                                           # tune window 7 Nov: MdAE $40.8 -> $40.7 but no-prior $55.2 -> $56.4, within $100 79.4% -> 79.2%
AIRBNB_SEPT_VALUES = False                 # also use September-2026 values (prices, revenue, scores, licences): after the cutoff

# ---- API (US11, sec. 10 of design.md) -----------------------------------------------------
API_BASE = "https://api-ht.microburbs.com.au"
API_KEY_ENV = "MICROBURBS_API_KEY"          # official name (demo repo); MICROBURBS_KEY also accepted
API_MAX_RPS = 8
API_PAGE = 1000
API_TIMEOUT_S = 30
API_RETRIES = 5
API_FROM_DATE = "2000-01-01"
API_DEADLINE = "12:30"
API_GATE_SHARE = 0.80
# The 29 suburbs the hackathon key covers (from the API docs) - fallback if /v1/suburbs/list returns something else
API_SUBURBS = ["Alexandria", "Barangaroo", "Beaconsfield (NSW)", "Camperdown (NSW)", "Centennial Park (NSW)",
               "Chippendale", "Darlinghurst", "Darlington (Sydney - NSW)", "Dawes Point", "Elizabeth Bay",
               "Erskineville", "Eveleigh", "Forest Lodge", "Glebe (NSW)", "Haymarket", "Millers Point", "Moore Park",
               "Potts Point", "Pyrmont", "Redfern", "Rosebery (NSW)", "Rushcutters Bay", "Surry Hills", "Sydney",
               "The Rocks (Sydney - NSW)", "Ultimo", "Waterloo (NSW)", "Woolloomooloo", "Zetland"]
API_ALLOW = ["/v1/me", "/v1/ping", "/v1/geocode/", "/v1/suburbs/list",
             "/listings/rent-history", "/rent-history", "/transport/nearby", "/market/median-rent-series"]
API_BLOCK = ["/listings/for-rent", "/listings/for-sale", "/rent-history-latest", "/market/rent-map",
             "/market/yield-map", "/market/median-rent", "/vacancy-rate", "/basics/", "/street-forecasts",
             "/valuation/"]
LATE_API_FROM = "2026-07-01"               # API-only adverts from here are tested in/out (T11.9)
LATE_API = "keep"                          # "keep" or "drop" - set from eval output

# ---- folders (outside the repo by default) -------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
CACHE_DIR = ROOT.parent / "cache" / "api"
OUT_DIR = ROOT / "out"
PUBLIC_DIR = ROOT / "public"                                                    # slim files from import_public.py
PUBLIC_SRC = ROOT.parent.parent / "microburbs-hackathon-2026" / "public-data" / "_derived"   # the team repo
LOOKUP_BAND = 100

# ---- lease file column aliases (T11.4) -----------------------------------------------------
LEASE_ALIASES = {
    "gnaf_id": ["gnaf_id", "gnaf_pid", "address_id"],
    "address": ["address", "full_address", "street_address", "property_address"],
    "building_address": ["building_address", "building", "building_street_address"],
    "suburb": ["suburb", "locality", "area_name"],
    "advert_date": ["advert_date", "listing_date", "listed_date", "date_listed", "date", "first_seen", "last_seen"],
    "rent": ["rent", "weekly_rent", "rent_pw", "asking_rent", "price", "rent_per_week"],
    "bedrooms": ["bedrooms", "beds"],
    "bathrooms": ["bathrooms", "baths"],
    "car_spaces": ["car_spaces", "parking", "cars"],
    "car_space": ["car_space", "has_car_space"],
    "lat": ["lat", "latitude"],
    "lng": ["lng", "lon", "longitude"],
}
LEASE_DROP = ["leased"]                    # outcome known after the advert date - never used

FLAT_COLUMNS = ["gnaf_id", "address", "building_address", "suburb", "mb", "latitude", "longitude",
                "property_type", "bedrooms", "bathrooms", "car_space", "car_spaces", "floor_level",
                "flats_in_building", "block_new_homes_2021_2026", "block_homes_2026",
                "ever_advertised_for_lease", "first_ever_listed", "attrs_last_seen"]
FLAT_REQUIRED = ["gnaf_id", "address", "latitude", "longitude"]
