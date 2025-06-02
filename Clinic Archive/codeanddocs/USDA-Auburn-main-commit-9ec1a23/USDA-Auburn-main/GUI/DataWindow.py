from PyQt6.QtCharts import * 
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from math import sin
from Settings import Settings

class DataWindow(QChartView):
    def __init__(self, epgdata):
        super().__init__()
        self.epgdata = epgdata
        self.transition_lines = []
        # We can probably update these to be more aesthetic.
        self.area_upper_lines = []
        self.area_lower_lines = []
        self.areas = []
        self.labels = []
        self.durations = []
        self.transition_lines = []
        self.track = False
        self.tracked_transition = -1
        self.transition_buffer = 0.1 # sec
        self.vertical_mode = False
        self.zoom_mode = False
        # self.h_scrollbar = QScrollBar()
        self.last_h_scroll_value = 0
        self.transition_mode = 'labels'
        self.file = None
        self.prepost = 'pre'
        self.critica_areas = []
        self.critical_state = False
        self.cursor_state = 0
        self.baseline = None
        self.initUI()
    
    def initUI(self):
        self.width = 400
        self.height = 400
        self.setGeometry(0, 0, self.width, self.height)

        # Create a series to hold our data
        self.series = QLineSeries()
        # Placeholder sine wave
        for i in range(10000):
            self.series.append(QPointF(i, sin(2*3.14 * i/ 10000)))

        # Create a chart to put our data in, put it on the chart
        self.chart = QChart()
        self.chart.setPlotAreaBackgroundVisible(True)
        self.chart.setPlotAreaBackgroundBrush(QBrush(Qt.GlobalColor.white))
        self.chart.setPlotAreaBackgroundPen(QPen(Qt.GlobalColor.black, 3))
        self.chart.setTitle("<b>SCIDO Waveform Editor</b>")
        self.chart.legend().hide()
        self.chart.addSeries(self.series)

        # Axes
        self.x_axis = QValueAxis()
        self.x_axis.setTitleText("Time [s]")
        self.x_axis.setGridLineVisible(Settings.show_grid)
        self.x_axis.setTickInterval(Settings.h_maj_gridline_spacing)
        self.x_axis.setTickAnchor(0)
        #self.x_axis.setLabelFormat("%.1f")  # Format the labels
        self.x_axis.setMinorTickCount(5) # Number of ticks between major ticks
        self.x_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.x_axis)
        #self.cursor.attachAxis(self.x_axis)
        
        self.y_axis = QValueAxis()
        self.y_axis.setTitleText("Volts")
        self.y_axis.setGridLineVisible(Settings.show_grid)
        self.y_axis.setTickInterval(Settings.v_maj_gridline_spacing)
        self.y_axis.setTickAnchor(0)
        #self.y_axis.setLabelFormat("%.1f")  # Format the labels
        self.y_axis.setMinorTickCount(5)  # Remove minor ticks if needed
        self.y_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.y_axis)
        
        # Set the chart of this QChartView to be our chart
        self.setChart(self.chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Add a readout of the compression level
        self.compression = 0
        self.compression_text = QGraphicsTextItem()
        self.compression_text.setDefaultTextColor(QColor(0, 0, 0))
        self.compression_text.setPos(100, 10)
        self.scene().addItem(self.compression_text)
        self.update_compression()

        # self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        # self.h_scrollbar.valueChanged.connect(lambda value: self.scroll_horizontal(value))
        # self.update_scrollbar()
        
    def resizeEvent(self, event):
        # resizeEvent is called automatically when the window is
        # resized and handles rescaling everything. We just also
        # want to update compression so we do that and then
        # let everything get handled normally.
        QChartView.resizeEvent(self, event) 
        self.update_compression()
        # self.update_scrollbar()

    def window_to_chart(self, x, y):
        """
        window_to_chart converts from window coordinates, like
        those from a mouse click event, to chart coordinates.
        Inputs:
            x, y: x and y coordinate of the window coordinate
        Returns:
            (chart_x, chart_y): chart coordinates equivalent to 
            the window coordinates.
        """
        scene_coords = self.chart.mapToScene(x, y)
        chart_coords = self.chart.mapFromScene(scene_coords)
        val_coords = self.chart.mapToValue(chart_coords)
        return (val_coords.x(), val_coords.y())

    def chart_to_window(self, x, y):
        """
        chart_to_window converts from chart coordinates to window
        coordinates, like those of a mouse click even.
        Inputs:
            x, y: x and y coordinates in chart coordinates
        Returns:
            (window_x, window_y): window coordinates equivalent
            to the chart coordinates.
        """
        chart_coords = self.chart.mapToPosition(QPointF(x, y))
        scene_coords = self.chart.mapToScene(chart_coords)
        window_coords = self.mapFromScene(scene_coords)
        return (window_coords.x(), window_coords.y())
        
    def updateCursor(self, event):
        return
        # Update the cursor line
        if not self.track:
            return
        x = event.pos().x()
        y = event.pos().y()
        scene_coords = self.mapToScene(x, y)
        chart_coords = self.chart.mapFromScene(scene_coords)
        val_coords = self.chart.mapToValue(chart_coords)
        self.cursor_x = val_coords.x()
        self.cursor.clear()
        self.cursor.append(QPointF(self.cursor_x, 0))
        self.cursor.append(QPointF(self.cursor_x, 1))

        # Update the cursor text
        hours = self.cursor_x // 3600
        minutes = (self.cursor_x - hours * 3600) // 60
        seconds = self.cursor_x - (hours * 3600) - (minutes * 60)
        self.cursortext.setPlainText(f"{hours :0>2.0f}:{minutes :0>2.0f}:{seconds :.2f}")
        self.cursortext.setPos(QPointF(x, 10))
    
    def update_compression(self):
        """
        update_compression updates the compression readout
        based on the zoom level according to the formula
        COMPRESSION = (SECONDS/PIXEL) * 125
        obtained by experimentation with WINDAQ. Note that 
        WINDAQ also has 'negative' compression levels for
        high levels of zooming out. We do not implement those here.
        Inputs:
            None
        Outputs:
            None
        """
        # Update the compression readout.
        # Get the pixel distance of one second, we use a wide range to avoid rounding issues.
        width = 1000
        pix_per_second = (self.chart_to_window(width, 0)[0] - \
                  self.chart_to_window(0, 0)[0]) / width
        second_per_pix = 1 / (pix_per_second)
        # Convert to compression based on WinDaq
        self.compression = second_per_pix * 125
        self.compression_text.setPlainText(f"Compression Level: {self.compression :.1f}")

        
    def get_closest_transition(self, event):
        """
        get_closest_transition takes in a click event from Qt 
        and returns the index of the transition that is closest
        to it along with the distance between the two in window
        coordinates (pixels)
        Inputs:
            event: a click event from Qt
        Returns:
            transition_index: the index of the nearest transition
            window_distance: the distance between x and the 
                     nearest transition in window units
        """
        click_x, _ = self.window_to_chart(event.pos().x(), event.pos().y())
        # This can be done faster but the lists are
        # short enough here it doesn't matter
        closest = float('inf')
        transition_index = None
        for i in range(len(self.transitions)):
            time = self.transitions[i][0]
            label = self.transitions[i][1]
            if abs(click_x - time) <= abs(click_x - closest):
                closest = time
                transition_index = i

        transition_x, _ = self.chart_to_window(closest, 0)
        window_distance = abs(transition_x - event.pos().x())
        
        return transition_index, window_distance

    
    def handle_labels(self, event):
        """
        handle_labels takes in a mouse event and then handles 
        updating labels with a dialog.
        Inputs:
            event: a mouse event generated by Qt.
        Returns:
            Nothing
        """
        # Check if click occurred in region with data
        click_x, _ = self.window_to_chart(event.pos().x(), event.pos().y())
        max_time = max(self.epgdata.get_recording(self.file, self.prepost)['time'])
        if click_x < 0 or click_x > max_time:
            return # outside of labeled region

        items = list(Settings.label_to_color.keys())
        item, ok = QInputDialog.getItem(self, "Change Waveform Type", "Waveform Type:", items, 0, False)
        if ok: # The user hit OK and wants to update the label
            # Find what region the user clicked in
            # This could be done faster with binary search
            # But it isn't necessary as of right now.
            transition_index = None
            for i in range(len(self.transitions)):
                if click_x < self.transitions[i][0]:
                    transition_index = i - 1
                    break
            if not transition_index:
                transition_index = len(self.transitions) - 1
            # Update transitions list and save
            self.transitions[transition_index] = (self.transitions[transition_index][0], item)
            print(f"transition mode: {self.transition_mode}")
            self.epgdata.set_transitions(self.file, self.transitions, self.transition_mode)
            # Change color and text label of rectangle accordingly
            self.areas[transition_index].setColor(Settings.label_to_color[item])
            self.labels[transition_index].setPointLabelsFormat(item)
    
    def add_drop_transitions(self, event):
        """
        add_drop_transitions takes in a mouse event and then handles
        either adding or dropping a transition with a dialog
        Inputs:
            event: a mouse event generated by Qt
        Returns:
            Nothing
        """
        # Check if click occurred in region with data
        click_x, _ = self.window_to_chart(event.pos().x(), event.pos().y())
        max_time = max(self.epgdata.get_recording(self.file, self.prepost)['time'])
        if click_x < 0 or click_x > max_time:
            return # outside of labeled region
        
        click_x = round(click_x, 2) # we only work in 100ths
        # Find the nearest transition. If it is close enough,
        # generate a dialog to confirm that we want to delete it
        transition_index, window_distance = self.get_closest_transition(event)
        items = list(Settings.label_to_color.keys())
        if window_distance < 10: # pixels
            if transition_index != 0: # don't allow deleting first transition
                delete = QMessageBox.question(self, "Modify Transition", "Delete Transition?")
                if delete == QMessageBox.StandardButton.Yes:
                    del self.transitions[transition_index]
                    # Remove the shaded area for this transition
                    self.chart.removeSeries(self.areas[transition_index])
                    del self.areas[transition_index]
                    del self.area_upper_lines[transition_index]
                    del self.area_lower_lines[transition_index]
                    # Remove the text label for this transition
                    self.chart.removeSeries(self.labels[transition_index])
                    del self.labels[transition_index]
                    del self.durations[transition_index]
                    # Make the transition before the one we deleted longer 
                    df = self.epgdata.get_recording(self.file, self.prepost)
                    next_transition = None
                    # Handle the case where we delete the
                    # last transition
                    if transition_index >= len(self.transitions):
                        next_transition = max(df['time'].values)
                    else:
                        next_transition = self.transitions[transition_index][0]
                    b_upper = self.area_upper_lines[transition_index - 1]
                    b_lower = self.area_lower_lines[transition_index - 1]
                    for line in [b_upper, b_lower]:
                        line_points = line.points()
                        line_points[-1].setX(next_transition)
                        line.replace(line_points)
                    # Recenter the label before this one
                    label_point = self.labels[transition_index - 1].points()
                    duration_point = self.durations[transition_index - 1].points()
                    midpoint = (self.transitions[transition_index - 1][0] + next_transition) / 2
                    label_point[-1].setX(midpoint)
                    duration_point[-1].setX(midpoint)
                    self.labels[transition_index - 1].replace(label_point)
                    self.durations[transition_index - 1].replace(duration_point)

                    # Remove the transition line for this transition
                    self.chart.removeSeries(self.transition_lines[transition_index])
                    del self.transition_lines[transition_index]

        # If there was no transition close enough, generate a dialog
        # to create a new one with a given label
        else:
            last_transition_index = len(self.transitions) - 1
            for i in range(len(self.transitions)):
                if click_x < self.transitions[i][0]:
                    last_transition_index = i - 1
                    break

            label, ok = QInputDialog.getItem(self, "Create New Transition", "Waveform type:", items, 0, False)
            if ok: # user clicked ok
                new_transition = (click_x, label)
                self.transitions.insert(last_transition_index + 1, new_transition)
                # Move the shaded area for the previous transition back
                b_upper = self.area_upper_lines[last_transition_index]
                b_lower = self.area_lower_lines[last_transition_index]
                for line in [b_upper, b_lower]:
                    line_points = line.points()
                    line_points[-1].setX(self.transitions[last_transition_index + 1][0])
                    line.replace(line_points)

                label_point = self.labels[last_transition_index].points()
                duration_point = self.durations[last_transition_index].points()
                midpoint = (self.transitions[last_transition_index + 1][0] + self.transitions[last_transition_index][0]) / 2
                label_point[-1].setX(midpoint)
                duration_point[-1].setX(midpoint)
                self.labels[last_transition_index].replace(label_point)
                self.durations[last_transition_index].replace(duration_point)

                # Add in a shaded area and text label for this transition
                df = self.epgdata.get_recording(self.file, self.prepost)
                volts = df[self.prepost + self.epgdata.prepost_suffix].values
                max_volts = max(volts)
                min_volts = min(volts)
                next_transition_time = None
                if last_transition_index + 2 >= len(self.transitions):
                    next_transition_time = max(df['time'].values)
                else:
                    next_transition_time = self.transitions[last_transition_index + 2][0]
                duration = next_transition_time - click_x
                upper_line = QLineSeries()
                upper_line.replace(
                    [QPointF(click_x, max_volts),
                     QPointF(click_x + duration, max_volts)]
                )
                lower_line = QLineSeries()
                lower_line.replace(
                    [QPointF(click_x, min_volts),
                     QPointF(click_x + duration, min_volts)]
                )
                area = QAreaSeries()
                area.setUpperSeries(upper_line)
                area.setLowerSeries(lower_line)
                self.chart.addSeries(area)
                area.attachAxis(self.x_axis)
                area.attachAxis(self.y_axis)
                area.setColor(Settings.label_to_color[label])
                area.setBorderColor(QColor(255, 255, 255, 0))
                
                self.area_upper_lines.insert(last_transition_index + 1, upper_line)
                self.area_lower_lines.insert(last_transition_index + 1, lower_line)
                self.areas.insert(last_transition_index + 1, area)


                label_series = QScatterSeries()
                label_series.setMarkerSize(1)
                label_series.setPointLabelsFont(QFont("Sans, 12, QFont.Bold"))
                label_series.setPointLabelsVisible(True)
                label_series.setPointLabelsFormat(label)
                self.chart.addSeries(label_series)
                label_y = (max_volts - min_volts) * 0.05 + min_volts
                label_series.append((next_transition_time - duration) + duration/2, label_y)
                label_series.attachAxis(self.x_axis)
                label_series.attachAxis(self.y_axis)
                self.labels.insert(last_transition_index + 1, label_series)

                duration_series = QScatterSeries()
                duration_series.setMarkerSize(1)
                duration_series.setPointLabelsFont(QFont("Sans, 12, QFont.Bold"))
                duration_series.setPointLabelsVisible(True)
                duration_series.setPointLabelsFormat(str(round(duration, 2)))
                self.chart.addSeries(duration_series)
                duration_y = (max_volts - min_volts) * 0.05 + min_volts - 10
                duration_series.append((next_transition_time - duration) + duration/2, duration_y)
                duration_series.attachAxis(self.x_axis)
                duration_series.attachAxis(self.y_axis)
                self.durations.insert(last_transition_index + 1, duration_series)

                # Add in a transition line for this transition
                vline = QLineSeries()
                vline.setColor(QColor(0, 0, 1))
                vline.append(QPointF(click_x, max_volts))
                vline.append(QPointF(click_x, min_volts))
                self.chart.addSeries(vline)
                vline.attachAxis(self.x_axis)
                vline.attachAxis(self.y_axis)
                self.transition_lines.insert(last_transition_index + 1, vline)

                # Remove and readd the transition line for 
                # the next transition, if it exists
                # self.chart.removeSeries(
        

        # Save transitions
        print(f"transition mode: {self.transition_mode}")
        self.epgdata.set_transitions(self.file, self.transitions, self.transition_mode)

    def handle_transitions(self, event, event_type):
        """
        handle_transitions takes in a click event and the event type
        and then handles interactive transition moving accordingly.
        Inputs:
            event: a mouse event generated by Qt.
            event_type: a string containing what type of mouse
                    event generated the event. This is probably
                    not needed but it makes implementation easy.
        Returns:
            Nothing
        """
        if event_type == "press":
            # Find nearest transition
            transition_index, window_distance = self.get_closest_transition(event)
            if transition_index == 0:
                return # Don't allow moving first transition
            elif window_distance < 10: #pixels
                self.tracked_transition = transition_index
            else:
                self.tracked_transition = -1
        elif event_type == "move" and self.tracked_transition >= 0:
            # Figure out where to move everything
            move_x, _  = self.window_to_chart(
                        event.pos().x(), 
                        event.pos().y())
            # Only allow movement between other transitions,
            # keep a buffer between them.
            prev_transition = self.transitions[self.tracked_transition - 1][0] + self.transition_buffer
            max_time = max(self.epgdata.get_recording(self.file, self.prepost)['time'])
            next_transition = None
            if self.tracked_transition == len(self.transitions) - 1:
                next_transition = max_time
            else:
                next_transition = self.transitions[self.tracked_transition + 1][0] - self.transition_buffer
            if move_x > prev_transition and move_x < next_transition: 
                # Redraw the areas before and after the transition
                b_upper = self.area_upper_lines[self.tracked_transition - 1]
                b_lower = self.area_lower_lines[self.tracked_transition - 1]
                for line in [b_upper, b_lower]:
                    line_points = line.points()
                    line_points[-1].setX(move_x)
                    line.replace(line_points)
                a_upper = self.area_upper_lines[self.tracked_transition]
                a_lower = self.area_lower_lines[self.tracked_transition]
                for line in [a_upper, a_lower]:
                    line_points = line.points()
                    line_points[0].setX(move_x)
                    line.replace(line_points)   
                # Redraw the transition line to be under the cursor
                transition_line = self.transition_lines[self.tracked_transition]
                line_points = transition_line.points()
                for point in line_points:
                    point.setX(move_x)
                transition_line.replace(line_points)
                # Update the entry in transitions
                self.transitions[self.tracked_transition] = (move_x, self.transitions[self.tracked_transition][1])

                # Update labels accordingly
                # Label after transition
                label_point = self.labels[self.tracked_transition].points()
                midpoint = (self.transitions[self.tracked_transition][0] + next_transition) / 2
                label_point[-1].setX(midpoint)
                self.labels[self.tracked_transition].replace(label_point)
                # Label before transition
                label_point = self.labels[self.tracked_transition - 1].points()
                midpoint = (self.transitions[self.tracked_transition - 1][0] + self.transitions[self.tracked_transition][0]) / 2
                label_point[-1].setX(midpoint)
                self.labels[self.tracked_transition - 1].replace(label_point)

                # TODO(MA): refactor by setting variables & look out 
                # for similarly repetitive/long reference sections VVV

                # update durations
                duration_point = self.durations[self.tracked_transition].points()
                midpoint = (self.transitions[self.tracked_transition][0] + next_transition) / 2
                duration_point[-1].setX(midpoint)
                self.durations[self.tracked_transition].replace(duration_point)
                self.durations[self.tracked_transition].setPointLabelsFormat(str(round(next_transition - self.transitions[self.tracked_transition][0], 2)))
                # duration before transition
                duration_point = self.durations[self.tracked_transition - 1].points()
                midpoint = (self.transitions[self.tracked_transition - 1][0] + self.transitions[self.tracked_transition][0]) / 2
                duration_point[-1].setX(midpoint)
                self.durations[self.tracked_transition - 1].replace(duration_point)
                self.durations[self.tracked_transition - 1].setPointLabelsFormat(str(round(self.transitions[self.tracked_transition][0] - self.transitions[self.tracked_transition - 1][0], 2)))
                

        elif event_type == "release" and self.tracked_transition >= 0:
            # Push transitions to EPGData
            print(f"transition mode: {self.transition_mode}")
            self.epgdata.set_transitions(self.file, self.transitions, self.transition_mode)
            # Reset tracking variables
            self.tracked_transition = -1

    def set_baseline(self, event: QMouseEvent):
        _, y = self.window_to_chart(event.pos().x(), event.pos().y())
        if self.baseline == None:
            self.baseline = QLineSeries()
            self.baseline.setColor(Qt.GlobalColor.gray)
            self.chart.addSeries(self.baseline)
            pen = self.baseline.pen()
            pen.setWidth(1)
            self.baseline.setPen(pen)
            self.baseline.attachAxis(self.x_axis)
            self.baseline.attachAxis(self.y_axis)
            # Generate the line
            pass
        self.baseline.clear()
        self.baseline.append(QPointF(self.x_axis.min(), y))
        self.baseline.append(QPointF(self.x_axis.max(), y))
        return
    
    def delete_baseline(self):
        if self.baseline == None:
            return
        self.baseline.clear()
        self.chart.removeSeries(self.baseline)
        self.baseline = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Shift: 
            # holding shift allows for vertical zoom / scroll
            self.vertical_mode = True
        elif event.key() == Qt.Key.Key_Control:
            # holding control allows for zoom, otherwise
            # scrolling scrolls the plot
            self.zoom_mode = True
        elif event.key() == Qt.Key.Key_R:
            # r resets zoom
            self.resetView()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Shift:
            self.vertical_mode = False
        elif event.key() == Qt.Key.Key_Control:
            self.zoom_mode = False
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.cursor_state == 0:
                self.handle_transitions(event, "press")
            else:
                self.set_baseline(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.add_drop_transitions(event)
    
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_labels(event)

    def mouseMoveEvent(self, event):
        self.handle_transitions(event, "move")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_transitions(event, "release")
    
    def updateBaseline(self):
        if self.baseline != None:
            y = self.baseline.at(0).y()
            self.baseline.clear()
            self.baseline.append(QPointF(self.x_axis.min(), y))
            self.baseline.append(QPointF(self.x_axis.max(), y))

    def scroll_horizontal(self, value):
        self.chart.scroll(value - self.last_h_scroll_value, 0)
        self.last_h_scroll_value = value
        self.updateBaseline()

    # def update_scrollbar(self):
    #     plot_area = self.chart.plotArea()

    #     self.h_scrollbar.setRange(int(-50000 / self.compression), int(50000 / self.compression))
    #     self.h_scrollbar.setPageStep(int(plot_area.width()) // 10)
            
    def zoom_scroll(self, event):
        """
        zoom_scroll is called every time there is a mouse wheel
        event and handles zooming and scrolling the plot
        Inputs:
            event: a wheel event
        Returns:
            Nothing
        """
        # Everything is mapped to vertical scrolling
        # to assume that their scroll wheel has a horizontal
        # scroller
        if self.zoom_mode:
            plot_area = self.chart.plotArea()
            delt = event.angleDelta().y()
            zoom_amount = delt
            if self.vertical_mode:
                self.chart.zoomIn(plot_area.adjusted(0, zoom_amount, 0, -1 * zoom_amount))
            else:
                self.chart.zoomIn(plot_area.adjusted(zoom_amount, 0, -1 * zoom_amount, 0))
                self.update_compression()
                # self.update_scrollbar()
                        
        else:
            if self.vertical_mode:
                self.chart.scroll(0, event.angleDelta().y())
            else:
                self.chart.scroll(event.angleDelta().y(), 0)
                self.updateBaseline()

    def resetView(self):
        self.chart.zoomReset()
        self.update_compression()

    def wheelEvent(self, event):
        """
        wheelEvent is called automatically whenever the scroll
        wheel is engaged over the chart. We use it to control
        horizontal and vertical scrolling along with zoom.
        """
        self.zoom_scroll(event)

    def set_comments_visible(self, visible):
        """
        Set comment visibility settings, and then plot/clear comments accordingly.
        Inputs:
            visible: whether to make comments visisble
        """
        Settings.show_comments = visible
        self.plot_comments()

    def plot_comments(self, clear_comments=False):
        """
        Plot comments ("event markers" in WinDAQ), if present.
        Inputs:  None
        Outputs: None
        """

        if hasattr(self, 'comment_lines'):
            for vline in self.comment_lines:
                self.chart.removeSeries(vline)
                self.comment_lines = []
        if hasattr(self, 'comment_textitems'):
            for comment_textitem in self.comment_textitems:
                self.scene().removeItem(comment_textitem)
                self.comment_textitems = []

        if not Settings.show_comments:
            return

        # NOTE: self.epgdata.get_recording only returns time and voltage columns (why?)
        # This needs the comment column as well, so access dataframe directly
        df = self.epgdata.dfs[self.file]

        if not 'comments' in df.columns:
            return
        comments_df = df[~df['comments'].isnull()]

        voltage   = df[self.prepost + self.epgdata.prepost_suffix].values
        min_voltage = min(voltage)
        max_voltage = max(voltage)
        comment_v = min_voltage + (max_voltage - min_voltage) * 0.1

        self.comments   = comments_df['comments'].values
        self.comments_t = comments_df['time'].values
        self.comments_v = [comment_v] * len(comments_df)

        self.comment_lines = []
        for time in self.comments_t:
            vline = QLineSeries()
            vline.setColor(QColor(0, 0, 0))
            pen = QPen()
            pen.setStyle(Qt.PenStyle.DashLine)
            vline.setPen(pen)
            vline.append(QPointF(time, min_voltage))
            vline.append(QPointF(time, max_voltage))
            self.chart.addSeries(vline)
            vline.attachAxis(self.x_axis)
            vline.attachAxis(self.y_axis)
            self.comment_lines.append(vline)

        # Wasn't able to figure out how to parent QGraphicsItem to chart directly,
        # passing QGraphicsTextItem(parent=self.chart) has no effect.
        # Ended up modifying code from https://forum.qt.io/post/546479,
        # but has issue that text appears on top of chart margins instead of going behind.

        self.comment_textitems = []
        for comment in self.comments:
            #comment_textitem = QGraphicsTextItem(comment)
            comment_textitem = QGraphicsTextItem()
            comment_textitem.setDefaultTextColor(QColor(0, 0, 0))
            comment_textitem.setHtml(f"<div style='background:rgba(255, 255, 255, 50%);'>{comment}</div>")
            comment_textitem.setRotation(270);
            #comment_textitem.setZValue(-1)
            self.comment_textitems.append(comment_textitem)
            self.chart.scene().addItem(comment_textitem)

        def update_comment_positions():
            for t, v, comment_textitem in zip(self.comments_t, self.comments_v, self.comment_textitems):
                point_data  = QPointF(t, v)
                point_chart = self.chart.mapToPosition(point_data, self.chart.series()[0])
                point_scene = self.chart.mapToScene(point_chart)
                # Alignment: https://stackoverflow.com/q/30037429
                point_scene.setX(point_scene.x() - comment_textitem.boundingRect().height()/2)
                comment_textitem.setPos(point_scene)

        self.chart.scene().changed.connect(update_comment_positions)

    def plot_recording(self, file, prepost = 'post'):
        """
        plot_recording creates a series for the time series given by file
        and updates the graph to show it.
        Inputs:
            file: a string containing the key of the recording
            prepost: a string containing either pre or post
                 to specify which recording is desired.
        Outputs:
            None
        """
        # Load data
        self.file = file
        df = self.epgdata.get_recording(self.file, prepost)
        time = df['time'].values
        volts = df[prepost + self.epgdata.prepost_suffix].values

        # Plot data
        points = [QPointF(t, v)  for t, v in zip(time, volts)]
        self.series.replace(points)
        
        # Update axis
        self.x_axis.setRange(min(time), max(time))
        self.y_axis.setRange(min(volts), max(volts))

    def plot_transitions(self, file, prepost = 'post'):
        """
        plot_transition creates a vertical line for each transition and
        then colors the region after it and before the next transition
        accordingly.
        Inputs:
            file: a string containing the key of the recording
        Outputs:
            None
        """

        # Color in areas, but delete old ones first, if there are any
        for area in self.areas:
            self.chart.removeSeries(area)

        for transition in self.transition_lines:
            self.chart.removeSeries(transition)

        for label in self.labels:
            self.chart.removeSeries(label)

        for duration in self.durations:
            self.chart.removeSeries(duration)

        #Load data
        self.file = file
        self.prepost = prepost
        df = self.epgdata.get_recording(self.file, prepost)
        volts = df[prepost + self.epgdata.prepost_suffix].values
        min_volts = min(volts)
        max_volts = max(volts)
        self.transitions = self.epgdata.get_transitions(self.file, self.transition_mode)
        print(f"transition mode: {self.transition_mode}")
        self.epgdata.set_transitions(self.file, self.transitions, self.transition_mode)

        # Only continue if the label column contains labels
        if self.epgdata.dfs[file][self.transition_mode].isna().all():
            return
        
        # try:
        durations = []
        for i in range(len(self.transitions) - 1):
            time, label = self.transitions[i]
            next_time, _ = self.transitions[i + 1]
            durations.append((time, next_time - time, label))
        durations.append((self.transitions[-1][0], max(df['time']), self.transitions[-1][1]))

        self.areas = []
        self.area_upper_lines = []
        self.area_lower_lines = []
        self.labels = []
        self.durations = []

        for (time, dur, label) in durations:
            area = QAreaSeries()
            if self.transition_mode == 'labels':
                 area.setColor(Settings.label_to_color[label])
            if self.transition_mode == 'probes':
                area.setColor(Settings.label_to_color[label])
            # Make borders see-through
            area.setBorderColor(QColor(255, 255, 255, 255))
            # We need to save the lines so that they
            # don't go out of scope at the end of this block.
            # If we don't then Qt will crash and won't tell
            # us why :( This is probably because of the c++
            # bindings...
            # Define upper line
            self.area_upper_lines.append(QLineSeries())
            self.area_upper_lines[-1].replace(
                [QPointF(time, max_volts),
                QPointF(time + dur, max_volts)])
            # Define lower line
            self.area_lower_lines.append(QLineSeries())
            self.area_lower_lines[-1].replace(
                [QPointF(time, min_volts),
                QPointF(time + dur, min_volts)])
            prev_transition = time
            # Define area and draw it
            area.setUpperSeries(self.area_upper_lines[-1])
            area.setLowerSeries(self.area_lower_lines[-1])
            self.chart.addSeries(area)
            area.attachAxis(self.x_axis)
            area.attachAxis(self.y_axis)
            self.areas.append(area)

        #Plot transitions, but delete old ones first
        for line in self.transition_lines:
            self.chart.removeSeries(line)

        self.transition_lines = []
        for time, label in self.transitions:
            vline = QLineSeries()
            vline.setColor(QColor(0, 0, 1))
            vline.append(QPointF(time, max_volts))
            vline.append(QPointF(time, min_volts))
            self.chart.addSeries(vline)
            vline.attachAxis(self.x_axis)
            vline.attachAxis(self.y_axis)
            self.transition_lines.append(vline)

        # Adding text labels
        for (time, dur, label) in durations:
            label_series = QScatterSeries()
            label_series.setMarkerSize(1)
            label_series.setPointLabelsFont(QFont("Sans, 12, QFont.Bold"))
            label_series.setPointLabelsVisible(True)
            label_series.setPointLabelsFormat(label)
            self.chart.addSeries(label_series)
            label_y = (max_volts - min_volts) * 0.05 + min_volts
            label_series.append(time + dur/2, label_y)
            label_series.attachAxis(self.x_axis)
            label_series.attachAxis(self.y_axis)
            self.labels.append(label_series)

            duration_series = QScatterSeries()
            duration_series.setMarkerSize(1)
            duration_series.setPointLabelsFont(QFont("Sans, 12, QFont.Bold"))
            duration_series.setPointLabelsVisible(False)
            duration_series.setPointLabelsFormat(str(round(dur, 2)))
            self.chart.addSeries(duration_series)
            duration_y = (max_volts - min_volts) * 0.8 + min_volts
            duration_series.append(time + dur/2, duration_y)
            duration_series.attachAxis(self.x_axis)
            duration_series.attachAxis(self.y_axis)
            self.durations.append(duration_series)

        # TODO: this try-catch is a **temporary** solution to clearing
        #       labels when an unlabeled file is loaded
        # except:
        #   for line in self.transition_lines:
        #       self.chart.removeSeries(line)
        #   raise(Exception("Could not find transitions."))

    # Slots for setting changes
    def change_label_color(self, label: str, color: QColor):
        """
        change_label_color is a slot for the signal emitted by the
        SettingsWindow on changing a label color.
        Inputs:
            label: label for the waveform background to recolor.
            color: color to change the label to.
        Returns:
            Nothing
        """
        for i in range(len(self.labels)):
            _, label_ = self.transitions[i]
            if label_ == label:
                self.areas[i].setColor(color)

    def change_line_color(self, color: QColor):
        """
        change_line_color is a slot for the signal emitted by the
        SettingsWindow on changing the line color.
        Inputs:
            color: color to which the recording line is to be changed
        Returns:
            Nothing
        """
        orig_color = self.series.color()
        self.series.setColor(color)

    def delete_label(self, label: str):
        """
        delete_label is a slot for the signal emitted by the
        SettingsWindow on deleting a label. The label dict is
        guaranteed updated at this point, so we must just update
        internal components to the DataWindow accordingly
        Inputs:
            label: label deleted
        Returns:
            Nothing
        """

    def set_gridlines(self, enable: bool):
        self.x_axis.setGridLineVisible(enable)
        self.x_axis.setMinorGridLineVisible(enable)
        self.y_axis.setGridLineVisible(enable)
        self.y_axis.setMinorGridLineVisible(enable)
        value = "enable" if enable else "disable"
        #print(f"DataWindow received call to {value} grid")

    def set_text_visible(self, visible: bool):
        for label in self.labels:
            label.setPointLabelsVisible(visible)

    def set_durations_visible(self, visible: bool):
        for duration in self.durations:
            duration.setPointLabelsVisible(visible)

    def set_h_gridline_spacing(self, value: float):
        self.x_axis.setTickInterval(value)
        #print(f"Horizontal gridline spacing set to {value}")

    def set_v_gridline_spacing(self, value: float):
        self.y_axis.setTickInterval(value)
        #print(f"Vertical gridline spacing set to {value}")

    def set_h_offset(self, value: float):
        self.x_axis.setTickAnchor(value)
        #print(f"Horizontal gridline offset set to {value}")

    def set_v_offset(self, value: float):
        self.y_axis.setTickAnchor(value)
        #print(f"Vertical gridline offset set to {value}")

    def hide_label(self, label: str, value: bool):
        to_show = Settings.labels_to_show[label] # We know a-priori that this function is only called after SettingsWindow has verified the existence of this label
        set_color = Settings.label_to_color[label]
        if value == True:
            set_color.setAlpha(Settings.alpha)
        else:
            set_color.setAlpha(0)
        print(f"Setting label {label} to show {to_show}")
        for i in range(len(self.transitions)):
            _, transition_label = self.transitions[i]
            if transition_label == label:
                self.areas[i].setColor(set_color)
