from re import match
import time
import math as m
import atexit
from datetime import datetime
from pioneer_sdk import Pioneer

class Drone():
    def __init__(self, ip:str='192.168.4.1', mavlink_port:str='8001', name:str='') -> None:
        self.__name = name
        if isinstance(ip, str):
            if match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
                self.__ip: str = ip
            else:
                raise SyntaxError("Given IP isn't valid.")
        else:
            raise TypeError("IP isn't a string.")
        if isinstance(mavlink_port, str):
            self.__mavlink_port: str = mavlink_port
        else:
            raise TypeError("Mavlink_Port isn't a string.")
        self.pioneer: Pioneer = Pioneer(self.__name, self.__ip, int(self.__mavlink_port), logger=False)
        self.controls = self.__Controls(self.pioneer)
        atexit.register(self.close_connection)
    
    @property
    def ip(self):
        return self.__ip
    
    @property
    def mavlink_port(self):
        return self.__mavlink_port
    
    def close_connection(self) -> None:
        self.pioneer.land(); self.pioneer.disarm()
        self.pioneer.close_connection()

    class __Controls:
        def __init__(self, drone: Pioneer) -> None:
            self.__drone: Pioneer = drone
            self.__points: list[dict] = []
            self.current_point_idx: int = -1
            self.current_rotation: float = 0.0
            self.__drone._point_reached = True
        
        @property
        def points(self):
            return self.__points
        
        def add_point(self, *args):
            for point in args:
                if all(_ in point for _ in ['x','y','z']):
                    self.__points.append(point) 
                else:
                    raise KeyError("Point doesn't have required keys.")
        
        def __get_next_point(self) -> dict:
            next_idx = (self.current_point_idx+1) % len(self.__points)
            return self.__points[next_idx]

        def __calculate_angle_to_point(self, point:dict) -> float:
            curr_coords = self.__drone.get_local_position_lps()
            if curr_coords is None:
                return 0.0
            cur_x,cur_y = curr_coords[0], curr_coords[1]
            tgt_x,tgt_y = point['x'],point['y']
            target_rotation = -m.atan2(tgt_y-cur_y,tgt_x-cur_x)+m.pi/2
            self.current_rotation = target_rotation
            return target_rotation

        def _fly_to_next_point(self) -> None:
            next_point = self.__get_next_point()
            x, y, z = tuple(next_point.get(key, 0) for key in ('x','y','z'))
            new_yaw = self.__calculate_angle_to_point(next_point)
            self.__drone.go_to_local_point(x,y,z,new_yaw)
            self.current_point_idx+=1

        def rotate_in_place(self, degrees: float = 360.0, deg_per_sec: float = 36.0) -> None:
            if degrees == 0:
                return
            angular_speed_rad = m.radians(deg_per_sec)
            duration = abs(degrees) / abs(deg_per_sec)  

            start = time.time()
            while time.time() - start < duration:
                self.__drone.set_manual_speed_body_fixed(0, 0, 0, angular_speed_rad)
            for _ in range(5):
                self.__drone.set_manual_speed_body_fixed(0, 0, 0, 0.0)

        def test(self):
            if not self.points:
                return
            drone = self.__drone
            drone.arm(); 
            while not drone.takeoff():
                pass
            while True:
                pr = drone.point_reached()
                if not pr:
                    continue
                self._fly_to_next_point()

if __name__=='__main__':
    d = Drone()
    d.controls.add_point(
        {"x":4.75, "y":0, "z":2},
        {"x":4.75, "y":4.75, "z":2},
        {"x":0, "y":4.75, "z":2},
        {"x":0, "y":0, "z":2},
    )
    d.controls.test()