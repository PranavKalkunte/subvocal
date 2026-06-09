"""Pydantic schemas for the Subvocal user context.

Defines the structure for contacts, calendar events, location, conversation,
and app state (including visible interactive elements).
"""


from pydantic import BaseModel, Field


class Contact(BaseModel):
    """Represents a user contact contact."""
    id: str = Field(description="Unique contact identifier")
    name: str = Field(description="Full name of the contact")
    phone: str | None = Field(default=None, description="Phone number")
    email: str | None = Field(default=None, description="Email address")
    relationship: str | None = Field(default=None, description="Relationship to user (e.g. spouse, manager)")
    shorthand_name: str = Field(description="Compressed phonetic shorthand of the contact's name")


class CalendarEvent(BaseModel):
    """Represents a calendar entry."""
    id: str = Field(description="Unique event identifier")
    title: str = Field(description="Calendar event title")
    start_time: str = Field(description="ISO 8601 start datetime string")
    end_time: str = Field(description="ISO 8601 end datetime string")
    location: str | None = Field(default=None, description="Event location")
    description: str | None = Field(default=None, description="Event description details")
    shorthand_title: str = Field(description="Compressed phonetic shorthand of the event title")


class LocationInfo(BaseModel):
    """Represents the user's current spatial coordinates and address."""
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    address: str | None = Field(default=None, description="Human-readable street address")
    place_name: str | None = Field(default=None, description="Name of the place (e.g. Home, Foundry Makerspace)")


class Message(BaseModel):
    """Represents a conversation turn in user-agent history."""
    role: str = Field(description="Role: 'user' or 'agent'")
    timestamp: float = Field(description="Epoch timestamp of message exchange")
    text: str = Field(description="Clean text content of message")


class UIElement(BaseModel):
    """Represents an interactive user interface element visible on screen."""
    element_id: str = Field(description="Unique CSS selector or element identifier")
    element_type: str = Field(description="Type of control (e.g. button, input, link, checkbox)")
    label: str = Field(description="Visible text label of the control")
    shorthand_label: str = Field(description="Compressed phonetic shorthand of the label")


class AppState(BaseModel):
    """Represents the active application and on-screen context."""
    current_app: str = Field(description="Active application name (e.g., Browser, Calendar, Messages)")
    page_url: str | None = Field(default=None, description="Current browser URL (if applicable)")
    page_title: str | None = Field(default=None, description="Current browser window or app page title")
    visible_elements: list[UIElement] = Field(default_factory=list, description="Interactive controls currently visible")


class UserContext(BaseModel):
    """Root model for the complete user context state."""
    contacts: list[Contact] = Field(default_factory=list, description="User's address book")
    calendar: list[CalendarEvent] = Field(default_factory=list, description="Upcoming calendar schedule")
    location: LocationInfo | None = Field(default=None, description="Current location information")
    conversation_history: list[Message] = Field(default_factory=list, description="Recent dialog turns")
    app_state: AppState = Field(default_factory=lambda: AppState(current_app=""), description="Current application and active UI elements")
