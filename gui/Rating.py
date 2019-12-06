import os

from PyQt5.QtCore import pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout


class RatingWidget(QWidget):
    value_updated = pyqtSignal(int)

    def __init__(self, parent=None):
        super(RatingWidget, self).__init__(parent)

        # Set defaults.
        self._value = 0

        # Dynamically create QWidget layout.
        hbox = QHBoxLayout()
        hbox.setSpacing(0)

        # Add icons to the layout.
        self.icons = []
        for icon_value in range(0, 5):
            icon_label = IconLabel(icon_value, parent=self)
            icon_label.mouse_enter_icon.connect(self._set_icons_visible)
            icon_label.mouse_leave_icon.connect(self._set_active_icons_visible)
            icon_label.mouse_release_icon.connect(self.set_icons_active)
            self.icons.append(icon_label)
            hbox.addWidget(icon_label)

        # Set the created layout to the widget.
        self.setLayout(hbox)
        self.installEventFilter(self)

    def _set_active_icons_visible(self):
        for icon in self.icons:
            icon.visible = icon.active

    def set_icons_active(self, icon_label):
        self._value = icon_label.value
        self.value_updated.emit(self._value)
        for icon in self.icons:
            icon.active = (icon.value <= icon_label.value)

    def _set_icons_visible(self, icon_label):
        for icon in self.icons:
            icon.visible = (icon.value <= icon_label.value)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Leave:
            self._set_active_icons_visible()
        else:
            super(RatingWidget, self).eventFilter(obj, event)
        return False

    @property
    def value(self):
        return self._value

    @property
    def max_value(self):
        return self._max_value


class IconLabel(QLabel):
    mouse_enter_icon = pyqtSignal(QLabel)
    mouse_leave_icon = pyqtSignal(QLabel)
    mouse_release_icon = pyqtSignal(QLabel)

    def __init__(self, value, parent=None):
        super(IconLabel, self).__init__(parent)
        self._active = False
        self._value = value

        # Enable mouse events without buttons being held down.
        self.setMouseTracking(True)
        self.setPixmap(QPixmap('gui/star0.png'))
        self.installEventFilter(self)

    def set_image(self, value):  # Set the image for the label.
        if value:
            self.setPixmap(QPixmap('gui/star1.png'))
        else:
            self.setPixmap(QPixmap('gui/star0.png'))

    def eventFilter(self, obj, event):  # Event filter defining custom actions.
        # When the mouse _enters_ the label area, set the icon visible.
        if event.type() == QEvent.Enter:
            self.mouse_enter_icon.emit(self)
        # When the mouse _leaves_ the label area, set the icon invisible.
        elif event.type() == QEvent.Leave:
            self.mouse_leave_icon.emit(self)
        # When the mouse _clicks_ the label area, set the icon active.
        elif event.type() == QEvent.MouseButtonRelease:
            self.mouse_release_icon.emit(self)
        else:
            super(IconLabel, self).eventFilter(obj, event)
        return False

    # Properties
    def _get_active(self):  # Get the active state of the label.
        return self._active

    def _set_active(self, value):  # Set the active state of the label.
        self._active = value

    def _get_value(self):  # Get the value state of the label.
        return self._value

    def _get_visible(self):  # Get the visible state of the label
        if not self.pixmap():
            return False
        else:
            return not self.pixmap().isNull()

    def _set_visible(self, value):  # Set the visible state of the label.
        self.set_image(value)

    active = property(_get_active, _set_active,
        doc="Get/Set the active state of the icon."
    )
    value = property(_get_value,
        doc="Get the value of the icon."
    )
    visible = property(_get_visible, _set_visible,
        doc="Get/Set the visible state of the icon."
    )

