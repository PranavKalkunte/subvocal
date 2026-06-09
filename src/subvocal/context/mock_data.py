"""Mock user context generator for testing and demonstration.

Generates structured mock data for contacts, calendar events, location,
conversation history, and active browser app state, automatically pre-computing
compressed phonetic shorthands for all text labels.
"""

import time

from subvocal.context.schema import AppState, CalendarEvent, Contact, LocationInfo, Message, UIElement, UserContext
from subvocal.shorthand.spec import compress_word


def generate_mock_context() -> UserContext:
    """Generate a high-fidelity UserContext populated with mock data."""
    
    # 1. Mock Contacts
    contacts_raw = [
        {"id": "c1", "name": "Alice Smith", "phone": "+1-512-555-0101", "email": "alice@gmail.com", "relationship": "Spouse"},
        {"id": "c2", "name": "Bob Jones", "phone": "+1-512-555-0102", "email": "bob.jones@work.com", "relationship": "Manager"},
        {"id": "c3", "name": "Charlie Brown", "phone": "+1-512-555-0103", "email": "charlie@brown.org", "relationship": "Friend"},
        {"id": "c4", "name": "Diana Prince", "phone": "+1-512-555-0104", "email": "diana@justice.org", "relationship": "Colleague"},
        {"id": "c5", "name": "Evan Wright", "phone": "+1-512-555-0105", "email": "evan.wright@tech.io", "relationship": "Colleague"}
    ]
    
    contacts = []
    for c in contacts_raw:
        # Generate shorthand name.
        # Compute compression for each token in name and join them.
        sh_parts = [compress_word(part) for part in c["name"].split()]
        sh_name = " ".join(sh_parts)
        contacts.append(Contact(
            id=c["id"],
            name=c["name"],
            phone=c["phone"],
            email=c["email"],
            relationship=c["relationship"],
            shorthand_name=sh_name
        ))
        
    # 2. Mock Calendar Events
    calendar_raw = [
        {"id": "e1", "title": "Team Sync Meeting", "start": "2026-06-08T10:00:00", "end": "2026-06-08T11:00:00", "loc": "Conference Room B", "desc": "Weekly status updates"},
        {"id": "e2", "title": "Lunch with Alice", "start": "2026-06-08T12:00:00", "end": "2026-06-08T13:00:00", "loc": "Corner Grill", "desc": "Personal lunch date"},
        {"id": "e3", "title": "BioSignals Research Review", "start": "2026-06-08T14:30:00", "end": "2026-06-08T15:30:00", "loc": "UT BioLab 3.12", "desc": "Discuss EMG-UKA dataset performance"}
    ]
    
    calendar = []
    for event in calendar_raw:
        sh_parts = [compress_word(part) for part in event["title"].split()]
        sh_title = " ".join(sh_parts)
        calendar.append(CalendarEvent(
            id=event["id"],
            title=event["title"],
            start_time=event["start"],
            end_time=event["end"],
            location=event["loc"],
            description=event["desc"],
            shorthand_title=sh_title
        ))
        
    # 3. Mock Location (UT Austin campus area)
    location = LocationInfo(
        latitude=30.2849,
        longitude=-97.7341,
        address="2501 Speedway, Austin, TX 78712",
        place_name="Foundry Makerspace"
    )
    
    # 4. Mock Conversation History
    history = [
        Message(role="user", timestamp=time.time() - 300, text="GOTO google.com"),
        Message(role="agent", timestamp=time.time() - 290, text="Opened google.com"),
        Message(role="user", timestamp=time.time() - 200, text="SEARCH weather forecast"),
        Message(role="agent", timestamp=time.time() - 190, text="Searching weather forecast on Google")
    ]
    
    # 5. Mock App State (Browser displaying Google Search results)
    elements_raw = [
        {"id": "el1", "type": "input", "label": "Search Box"},
        {"id": "el2", "type": "button", "label": "Google Search Button"},
        {"id": "el3", "type": "link", "label": "Sign In"},
        {"id": "el4", "type": "link", "label": "Settings"},
        {"id": "el5", "type": "button", "label": "Submit"}
    ]
    
    visible_elements = []
    for el in elements_raw:
        sh_parts = [compress_word(part) for part in el["label"].split()]
        sh_label = " ".join(sh_parts)
        visible_elements.append(UIElement(
            element_id=el["id"],
            element_type=el["type"],
            label=el["label"],
            shorthand_label=sh_label
        ))
        
    app_state = AppState(
        current_app="Browser",
        page_url="https://www.google.com",
        page_title="Google Search",
        visible_elements=visible_elements
    )
    
    # Assemble final context
    return UserContext(
        contacts=contacts,
        calendar=calendar,
        location=location,
        conversation_history=history,
        app_state=app_state
    )
