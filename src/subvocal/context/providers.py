"""Modular ContextProvider implementations for the Subvocal SDK.
"""


from subvocal.context.schema import (
    AppState,
    CalendarEvent,
    Contact,
    LocationInfo,
    UserContext,
)
from subvocal.core.interfaces import ContextProvider


class StaticContextProvider(ContextProvider):
    """Simple provider wrapping a static UserContext instance (useful for tests/mock runs)."""

    def __init__(self, context: UserContext, name: str = "static_context"):
        self.context = context
        self.name = name

    def get_context(self) -> UserContext:
        return self.context

    def get_provider_name(self) -> str:
        return self.name


class CalendarContextProvider(ContextProvider):
    """Modular provider for retrieving calendar events."""

    def __init__(self, events: list[CalendarEvent] | None = None):
        self.events = events or []

    def get_provider_name(self) -> str:
        return "calendar_provider"

    def get_context(self) -> UserContext:
        return UserContext(
            calendar=self.events,
            app_state=AppState(current_app="System")
        )


class ContactsContextProvider(ContextProvider):
    """Modular provider for retrieving user contact directory."""

    def __init__(self, contacts: list[Contact] | None = None):
        self.contacts = contacts or []

    def get_provider_name(self) -> str:
        return "contacts_provider"

    def get_context(self) -> UserContext:
        return UserContext(
            contacts=self.contacts,
            app_state=AppState(current_app="System")
        )


class LocationContextProvider(ContextProvider):
    """Modular provider for retrieving current location details."""

    def __init__(self, location: LocationInfo | None = None):
        self.location = location

    def get_provider_name(self) -> str:
        return "location_provider"

    def get_context(self) -> UserContext:
        return UserContext(
            location=self.location,
            app_state=AppState(current_app="System")
        )


class AppStateContextProvider(ContextProvider):
    """Modular provider for retrieving active app state and visible elements."""

    def __init__(self, app_state: AppState | None = None):
        self.app_state = app_state or AppState(current_app="Launcher")

    def get_provider_name(self) -> str:
        return "app_state_provider"

    def get_context(self) -> UserContext:
        return UserContext(
            app_state=self.app_state
        )


class CompositeContextProvider(ContextProvider):
    """Aggregates multiple individual ContextProviders into a single UserContext."""

    def __init__(self, providers: list[ContextProvider]):
        self.providers = providers

    def get_provider_name(self) -> str:
        return "composite_provider"

    def get_context(self) -> UserContext:
        merged_contacts = []
        merged_calendar = []
        merged_location = None
        merged_app_state = AppState(current_app="System")

        for provider in self.providers:
            ctx = provider.get_context()
            if ctx.contacts:
                merged_contacts.extend(ctx.contacts)
            if ctx.calendar:
                merged_calendar.extend(ctx.calendar)
            if ctx.location is not None:
                merged_location = ctx.location
            # The last active non-System app state takes priority
            if ctx.app_state.current_app != "System":
                merged_app_state = ctx.app_state

        return UserContext(
            contacts=merged_contacts,
            calendar=merged_calendar,
            location=merged_location,
            app_state=merged_app_state
        )
