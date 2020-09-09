class Commander(): 
    #One centralized commander/controller, and one commander/controller for each of the drones?
    #Centralized commander/controller sends messages to the controller of the specific drone?
    """
    Used for sending control setpoints to the CrazyFlie
    """

    def __init__(self, crazyflie = None):
        self._cf = crazyflie
    
    def send_setpoint(self, roll, pitch, yaw, thrust):
        """
        Send a new control setpoint for roll/pitch/yaw/thrust to the copter
        The arguments roll/pitch/yaw/trust is the new setpoints that should
        be sent to the copter
        """
        pass

    def send_position_setpoint(self, x, y, yaw): #2D, if 3D: add z
        """
        Control mode where the position is sent as absolute x,y coordinate in
        meter and the yaw is the absolute orientation.
        x and y are in m
        yaw is in degrees
        """
        pk = simPacket(simPacket.TYPE_POSITION, (x,y,yaw))
        self._cf.update_setpoints(pk)
        pass
    
    def send_packet(self, package):
        pass


class simPacket():
    TYPE_VELOCITY_WORLD = 0
    TYPE_POSITION = 1

    def __init__(self, _pkType: int, _data: tuple = None):
        self.pkType = _pkType
        self.data = _data
    
    def __repr__(self):
        return (f'Package type: {self.pkType}, Data: {self.data}')