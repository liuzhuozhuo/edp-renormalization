from manim import *
from manim_slides import Slide

class diagram_product(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        H1_text = MathTex("\mathcal{H}_{01}").set_color(BLACK).scale(2).move_to(LEFT)
        H1_text_copy = H1_text.copy().move_to(RIGHT)
        Prod = MathTex("\cdot").set_color(BLACK).scale(2)

        #Scene 1: H_1*H_1
        self.add(H1_text, H1_text_copy, Prod)
        self.next_slide()
        self.play(H1_text.animate.shift(UP*2.5), H1_text_copy.animate.shift(UP*2.5), Prod.animate.shift(UP*2.5))

        dot = Dot([0, 0, 0])
        dot2 = Dot([2, 0, 0])
        dot3 = Dot([4, 1, 0])
        dot4 = Dot([4, -1, 0])
        line = Line(dot.get_center(), dot2.get_center()).set_color(BLACK)
        line2 = Line(dot2.get_center(), dot3.get_center()).set_color(BLACK)
        line3 = Line(dot2.get_center(), dot4.get_center()).set_color(BLACK)

        diagram1 = VGroup(line, line2, line3).move_to(RIGHT*3)

        dot5 = Dot([4, 0, 0])
        dot6 = Dot([2, 0, 0])
        dot7 = Dot([0, 1, 0])
        dot8 = Dot([0, -1, 0])
        line4 = Line(dot5.get_center(), dot6.get_center()).set_color(BLACK)
        line5 = Line(dot6.get_center(), dot7.get_center()).set_color(BLACK)
        line6 = Line(dot6.get_center(), dot8.get_center()).set_color(BLACK)

        diagram2 = VGroup(line4, line5, line6).move_to(LEFT*3)

        plus_sign = MathTex("+").set_color(BLACK).move_to(ORIGIN).scale(2)
        parenthesis_l = MathTex("(").set_color(BLACK).move_to(LEFT*6).scale(5)
        parenthesis_r = MathTex(")").set_color(BLACK).move_to(RIGHT*6).scale(5)
        
        H1 = VGroup(diagram1, diagram2, plus_sign, parenthesis_l, parenthesis_r).scale(0.4).move_to(LEFT*3 + UP*2.5)
        diagram1_copy = diagram1.copy()
        diagram2_copy = diagram2.copy()
        plus_sign_copy = plus_sign.copy()
        parenthesis_l_copy = parenthesis_l.copy()
        parenthesis_r_copy = parenthesis_r.copy()
        H1_copy = VGroup(diagram1_copy, diagram2_copy, plus_sign_copy, parenthesis_l_copy, parenthesis_r_copy).move_to(RIGHT*3 + UP*2.5)

        diagram1_1 = diagram1.copy()
        diagram2_1 = diagram2.copy()
        diagram2_copy_1 = diagram2_copy.copy()
        diagram1_copy_1 = diagram1_copy.copy()

        self.play(Transform(H1_text, H1))
        self.play(Transform(H1_text_copy, H1_copy))

        self.next_slide()

        Prod_1 = MathTex("\cdot").set_color(BLACK).scale(1.5).move_to(LEFT*3)
        Prod_2 = MathTex("\cdot").set_color(BLACK).scale(1.5).move_to(RIGHT*2)
        Prod_3 = MathTex("\cdot").set_color(BLACK).scale(1.5).move_to(DOWN * 2 + LEFT*2)
        Prod_4 = MathTex("\cdot").set_color(BLACK).scale(1.5).move_to(DOWN * 2 + RIGHT*3)

        plus_sign_1 = MathTex("+").set_color(BLACK).scale(1).move_to(LEFT*0.5)
        plus_sign_2 = MathTex("+").set_color(BLACK).scale(1).move_to(RIGHT*4.5)
        plus_sign_3 = MathTex("+").set_color(BLACK).scale(1).move_to(DOWN*2 + LEFT*4.5)
        plus_sign_4 = MathTex("+").set_color(BLACK).scale(1).move_to(DOWN*2 + RIGHT*0.5)

        equal_1 = MathTex("=").set_color(BLACK).scale(1.5).move_to(UP*2.5 + RIGHT*6)
        equal_2 = MathTex("=").set_color(BLACK).scale(1.5).move_to(LEFT*5.5)

        self.play(equal_1.animate.add(), equal_2.animate.add())

        self.play(diagram2.animate.move_to(LEFT*4), diagram2_copy.animate.move_to(LEFT*2), Prod_1.animate.add(), plus_sign_1.animate.add(), run_time = 0.5)
        self.play(diagram1.animate.move_to(RIGHT), diagram2_copy_1.animate.move_to(RIGHT*3), Prod_2.animate.add(), plus_sign_2.animate.add(), run_time = 0.5)
        self.play(diagram2_1.animate.move_to(DOWN*2 + LEFT*3), diagram1_copy.animate.move_to(DOWN*2 + LEFT), Prod_3.animate.add(), plus_sign_3.animate.add(), run_time = 0.5)
        self.play(diagram1_1.animate.move_to(DOWN*2 + RIGHT*2), diagram1_copy_1.animate.move_to(DOWN*2 + RIGHT*4), Prod_4.animate.add(), plus_sign_4.animate.add(), run_time = 0.5)

        self.next_slide()
        self.clear()

        self.play(diagram2.animate.add(), diagram2_copy.animate.add(), run_time = 0.5)

        original1 = VGroup(diagram2.copy(), diagram2_copy.copy()).shift(RIGHT*5 + UP*2.5)

