"""Evaluation test dataset for Subvocal Shorthand reconstruction.

Contains 50 test cases covering navigation, interactions, typing, searches,
and system controls under various noisy environments.
"""

from typing import Any

EVAL_SET: list[dict[str, Any]] = [
    # --- Category 1: GOTO Navigation (10 cases) ---
    {
        "noisy": "gt ggl.cm",
        "expected": "GOTO google.com",
        "ui_elements": ["Google.com", "Gmail", "Google Maps"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to Google homepage"
    },
    {
        "noisy": "gt gthb.cm",
        "expected": "GOTO github.com",
        "ui_elements": ["GitHub", "Pull Requests", "Issues"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to GitHub"
    },
    {
        "noisy": "gt mail.google.com",
        "expected": "GOTO mail.google.com",
        "ui_elements": ["Inbox", "Sent Mail", "mail.google.com"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to Gmail"
    },
    {
        "noisy": "gt rddt.cm",
        "expected": "GOTO reddit.com",
        "ui_elements": ["Reddit", "Subreddits", "Popular"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to Reddit"
    },
    {
        "noisy": "gt yt.cm",
        "expected": "GOTO youtube.com",
        "ui_elements": ["YouTube", "Trending", "Subscriptions"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to YouTube"
    },
    {
        "noisy": "gt nbc.cm",
        "expected": "GOTO news.com",
        "ui_elements": ["News.com", "Latest News", "Weather"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to News site with consonant deletion error"
    },
    {
        "noisy": "gt wkpda.org",
        "expected": "GOTO wikipedia.org",
        "ui_elements": ["Wikipedia", "Wiki Page", "Search Wikipedia"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to Wikipedia"
    },
    {
        "noisy": "gt trhks.org",
        "expected": "GOTO treehacks.org",
        "ui_elements": ["TreeHacks", "Hacker Dashboard", "Sponsors"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to competitor page"
    },
    {
        "noisy": "gt loclhst.3000",
        "expected": "GOTO localhost:3000",
        "ui_elements": ["Localhost Dev", "Console", "Network"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to local development server"
    },
    {
        "noisy": "gt fcbk.cm",
        "expected": "GOTO facebook.com",
        "ui_elements": ["Facebook", "Friends", "Feed"],
        "contacts": [],
        "calendar": [],
        "description": "Navigate to Facebook"
    },

    # --- Category 2: CLICK / SCROLL / CLOSE Interactions (15 cases) ---
    {
        "noisy": "clk sbt",
        "expected": "CLICK Submit",
        "ui_elements": ["Submit", "Cancel", "Reset Form"],
        "contacts": [],
        "calendar": [],
        "description": "Click Submit button"
    },
    {
        "noisy": "clk sgn n",
        "expected": "CLICK Sign In",
        "ui_elements": ["Sign In", "Sign Up", "Forgot Password"],
        "contacts": [],
        "calendar": [],
        "description": "Click Sign In link"
    },
    {
        "noisy": "clk lgt",
        "expected": "CLICK Logout",
        "ui_elements": ["Profile Settings", "Logout", "Help"],
        "contacts": [],
        "calendar": [],
        "description": "Click Logout button"
    },
    {
        "noisy": "clk nxt",
        "expected": "CLICK Next",
        "ui_elements": ["Back", "Next Page", "Done"],
        "contacts": [],
        "calendar": [],
        "description": "Click Next step button"
    },
    {
        "noisy": "clk cncl",
        "expected": "CLICK Cancel",
        "ui_elements": ["Confirm Purchase", "Cancel", "Go Back"],
        "contacts": [],
        "calendar": [],
        "description": "Click Cancel link"
    },
    {
        "noisy": "clk ad t crt",
        "expected": "CLICK Add to Cart",
        "ui_elements": ["Add to Cart", "Buy Now", "Add to Wishlist"],
        "contacts": [],
        "calendar": [],
        "description": "Click Add to Cart button"
    },
    {
        "noisy": "clk ply btn",
        "expected": "CLICK Play Button",
        "ui_elements": ["Play Button", "Pause", "Volume Control"],
        "contacts": [],
        "calendar": [],
        "description": "Click media player play control"
    },
    {
        "noisy": "clk dwnld",
        "expected": "CLICK Download",
        "ui_elements": ["View Source", "Download PDF", "Share File"],
        "contacts": [],
        "calendar": [],
        "description": "Click Download PDF link"
    },
    {
        "noisy": "scrl dwn",
        "expected": "SCROLL Down",
        "ui_elements": ["Page Header", "Page Content"],
        "contacts": [],
        "calendar": [],
        "description": "Scroll viewport down"
    },
    {
        "noisy": "scrl p",
        "expected": "SCROLL Up",
        "ui_elements": ["Page Footer", "Top of Page"],
        "contacts": [],
        "calendar": [],
        "description": "Scroll viewport up"
    },
    {
        "noisy": "scrl lft",
        "expected": "SCROLL Left",
        "ui_elements": ["Carousel Image", "Nav Arrow"],
        "contacts": [],
        "calendar": [],
        "description": "Scroll viewport left"
    },
    {
        "noisy": "scrl rgt",
        "expected": "SCROLL Right",
        "ui_elements": ["Carousel Image", "Nav Arrow"],
        "contacts": [],
        "calendar": [],
        "description": "Scroll viewport right"
    },
    {
        "noisy": "cls tb",
        "expected": "CLOSE Tab",
        "ui_elements": ["New Tab", "Close Tab", "Bookmarks"],
        "contacts": [],
        "calendar": [],
        "description": "Close browser tab"
    },
    {
        "noisy": "cls wndw",
        "expected": "CLOSE Window",
        "ui_elements": ["Minimize", "Maximize", "Close Window"],
        "contacts": [],
        "calendar": [],
        "description": "Close desktop window"
    },
    {
        "noisy": "clk srh btn",
        "expected": "CLICK Search Button",
        "ui_elements": ["Search Input", "Search Button", "Advanced Search"],
        "contacts": [],
        "calendar": [],
        "description": "Click search submit control"
    },

    # --- Category 3: SEARCH browser/web queries (10 cases) ---
    {
        "noisy": "srch bsnl",
        "expected": "SEARCH BioSignals",
        "ui_elements": [],
        "contacts": [],
        "calendar": ["BioSignals Research Review", "Team Sync"],
        "description": "Search calendar topics"
    },
    {
        "noisy": "srch mcbn lrng",
        "expected": "SEARCH machine learning",
        "ui_elements": ["Machine Learning Wiki", "Deep Learning Tutorial"],
        "contacts": [],
        "calendar": [],
        "description": "Search scientific topic with consonant substitutions"
    },
    {
        "noisy": "srch wthr dstn",
        "expected": "SEARCH weather Austin",
        "ui_elements": ["Austin Weather Today", "Rain Forecast"],
        "contacts": [],
        "calendar": [],
        "description": "Search local weather"
    },
    {
        "noisy": "srch nrl ntwk",
        "expected": "SEARCH neural networks",
        "ui_elements": ["Neural Networks Journal", "AI Foundations"],
        "contacts": [],
        "calendar": [],
        "description": "Search AI concepts"
    },
    {
        "noisy": "srch fgt calb",
        "expected": "SEARCH Figma calibration",
        "ui_elements": ["Figma Calibration Specs", "Design System"],
        "contacts": [],
        "calendar": [],
        "description": "Search hardware/design files"
    },
    {
        "noisy": "srch emguka cp",
        "expected": "SEARCH EMG-UKA corpus",
        "ui_elements": ["EMG-UKA Corpus Download", "Academic Papers"],
        "contacts": [],
        "calendar": [],
        "description": "Search dataset specs"
    },
    {
        "noisy": "srch snt spch",
        "expected": "SEARCH silent speech",
        "ui_elements": ["Silent Speech Interface", "BCI Technology"],
        "contacts": [],
        "calendar": [],
        "description": "Search silent speech research"
    },
    {
        "noisy": "srch flyt scdl",
        "expected": "SEARCH flight schedule",
        "ui_elements": ["Book Flight", "Flight Status Tracker"],
        "contacts": [],
        "calendar": [],
        "description": "Search flight availability"
    },
    {
        "noisy": "srch stks tdy",
        "expected": "SEARCH stocks today",
        "ui_elements": ["Market Trends", "Stock Price Indices"],
        "contacts": [],
        "calendar": [],
        "description": "Search financial updates"
    },
    {
        "noisy": "srch dfn emg",
        "expected": "SEARCH definition EMG",
        "ui_elements": ["Electromyography", "EMG Medical Definition"],
        "contacts": [],
        "calendar": [],
        "description": "Search definition query"
    },

    # --- Category 4: TYPE Input (10 cases) ---
    {
        "noisy": "typ alc smth",
        "expected": "TYPE Alice Smith",
        "ui_elements": [],
        "contacts": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "calendar": [],
        "description": "Type spouse contact name"
    },
    {
        "noisy": "typ bb jns",
        "expected": "TYPE Bob Jones",
        "ui_elements": [],
        "contacts": ["Alice Smith", "Bob Jones", "Evan Wright"],
        "calendar": [],
        "description": "Type manager name"
    },
    {
        "noisy": "typ hl wrld",
        "expected": "TYPE hello world",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Type simple hello world greeting"
    },
    {
        "noisy": "typ ur srvc",
        "expected": "TYPE your service",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Type shorthand greeting phrase"
    },
    {
        "noisy": "typ ths s amzg",
        "expected": "TYPE this is amazing",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Type feedback message"
    },
    {
        "noisy": "typ psrd123",
        "expected": "TYPE password123",
        "ui_elements": ["Login Form", "Password Input"],
        "contacts": [],
        "calendar": [],
        "description": "Type numeric/text password string"
    },
    {
        "noisy": "typ usrnm_dev",
        "expected": "TYPE username_dev",
        "ui_elements": ["Username Input"],
        "contacts": [],
        "calendar": [],
        "description": "Type username input"
    },
    {
        "noisy": "typ emg rsrch",
        "expected": "TYPE EMG research",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Type domain specific research term"
    },
    {
        "noisy": "typ evn wrgt",
        "expected": "TYPE Evan Wright",
        "ui_elements": [],
        "contacts": ["Alice Smith", "Evan Wright", "Charlie Brown"],
        "calendar": [],
        "description": "Type coworker name"
    },
    {
        "noisy": "typ thx fr hlp",
        "expected": "TYPE thanks for help",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Type conversational response"
    },

    # --- Category 5: System Controls & Clipboard (5 cases) ---
    {
        "noisy": "cnfm",
        "expected": "CONFIRM",
        "ui_elements": ["Confirm Transaction", "Deny"],
        "contacts": [],
        "calendar": [],
        "description": "Single-word confirm control"
    },
    {
        "noisy": "cncl",
        "expected": "CANCEL",
        "ui_elements": ["Ok", "Cancel"],
        "contacts": [],
        "calendar": [],
        "description": "Single-word cancel control"
    },
    {
        "noisy": "und",
        "expected": "UNDO",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Revert last action"
    },
    {
        "noisy": "cpy",
        "expected": "COPY",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Copy command"
    },
    {
        "noisy": "pst",
        "expected": "PASTE",
        "ui_elements": [],
        "contacts": [],
        "calendar": [],
        "description": "Paste command"
    }
]
